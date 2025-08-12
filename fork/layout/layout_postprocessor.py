import os
import sys
import bisect
import logging
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.page import TextCell
from rtree import index

from docling.datamodel.base_models import BoundingBox, Page
from docling.datamodel.pipeline_options import LayoutOptions

from base_models import Cluster
from fork.timers import _CPUTimer

_log = logging.getLogger(__name__)


class UnionFind:
    """Efficient Union-Find data structure for grouping elements."""

    def __init__(self, elements):
        self.parent = {elem: elem for elem in elements}
        self.rank = dict.fromkeys(elements, 0)

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x == root_y:
            return

        if self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        elif self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def get_groups(self) -> Dict[int, List[int]]:
        """Returns groups as {root: [elements]}."""
        groups = defaultdict(list)
        for elem in self.parent:
            groups[self.find(elem)].append(elem)
        return groups


class SpatialClusterIndex:
    """Efficient spatial indexing for clusters using R-tree and interval trees."""

    def __init__(self, clusters: List[Cluster], build_intervals: bool = True):
        p = index.Property()
        p.dimension = 2
        self.spatial_index = index.Index(properties=p)
        self.build_intervals = build_intervals
        if build_intervals:
            self.x_intervals = IntervalTree()
            self.y_intervals = IntervalTree()
        self.clusters_by_id: Dict[int, Cluster] = {}

        for cluster in clusters:
            self.add_cluster(cluster)

    def add_cluster(self, cluster: Cluster):
        bbox = cluster.bbox
        self.spatial_index.insert(cluster.id, bbox.as_tuple())
        if self.build_intervals:
            self.x_intervals.insert(bbox.l, bbox.r, cluster.id)
            self.y_intervals.insert(bbox.t, bbox.b, cluster.id)
        self.clusters_by_id[cluster.id] = cluster

    def remove_cluster(self, cluster: Cluster):
        self.spatial_index.delete(cluster.id, cluster.bbox.as_tuple())
        del self.clusters_by_id[cluster.id]

    def find_candidates(self, bbox: BoundingBox) -> Set[int]:
        """Find potential overlapping cluster IDs using all indexes."""
        if not self.build_intervals:
            # Fallback: just spatial index
            return set(self.spatial_index.intersection(bbox.as_tuple()))
        
        spatial = set(self.spatial_index.intersection(bbox.as_tuple()))
        x_candidates = self.x_intervals.find_containing(
            bbox.l
        ) | self.x_intervals.find_containing(bbox.r)
        y_candidates = self.y_intervals.find_containing(
            bbox.t
        ) | self.y_intervals.find_containing(bbox.b)
        return spatial.union(x_candidates).union(y_candidates)

    def check_overlap(
            self,
            bbox1: BoundingBox,
            bbox2: BoundingBox,
            overlap_threshold: float,
            containment_threshold: float,
            epsilon: float = 0.0,
    ) -> bool:
        """Check if two bboxes overlap sufficiently."""
        if bbox1.area() <= 0 or bbox2.area() <= 0:
            return False

        iou = bbox1.intersection_over_union(bbox2)
        containment1 = bbox1.intersection_over_self(bbox2)
        containment2 = bbox2.intersection_over_self(bbox1)

        # Apply epsilon to thresholds to handle tie-breaks consistently
        effective_overlap_threshold = overlap_threshold - epsilon
        effective_containment_threshold = containment_threshold - epsilon

        return (
                iou >= effective_overlap_threshold
                or containment1 >= effective_containment_threshold
                or containment2 >= effective_containment_threshold
        )


class Interval:
    """Helper class for sortable intervals."""

    def __init__(self, min_val: float, max_val: float, id: int):
        self.min_val = min_val
        self.max_val = max_val
        self.id = id

    def __lt__(self, other):
        if isinstance(other, Interval):
            return self.min_val < other.min_val
        return self.min_val < other


class IntervalTree:
    """Memory-efficient interval tree for 1D overlap queries."""

    def __init__(self):
        self.intervals: List[Interval] = []  # Sorted by min_val

    def insert(self, min_val: float, max_val: float, id: int):
        interval = Interval(min_val, max_val, id)
        bisect.insort(self.intervals, interval)

    def find_containing(self, point: float) -> Set[int]:
        """Find all intervals containing the point."""
        pos = bisect.bisect_left(self.intervals, point)
        result = set()

        # Check intervals starting before point
        for interval in reversed(self.intervals[:pos]):
            if interval.min_val <= point <= interval.max_val:
                result.add(interval.id)
            else:
                break

        # Check intervals starting at/after point
        for interval in self.intervals[pos:]:
            if point <= interval.max_val:
                if interval.min_val <= point:
                    result.add(interval.id)
            else:
                break

        return result


class GridIndex:
    """Fast spatial grid for dense regular cluster overlap queries."""
    
    def __init__(self, page_w: float, page_h: float, bin_w: float, bin_h: float):
        # Guard rails
        self.bin_w = max(bin_w, 1e-6)
        self.bin_h = max(bin_h, 1e-6)
        self.nx = max(1, int(page_w / self.bin_w))
        self.ny = max(1, int(page_h / self.bin_h))
        self.cells: Dict[Tuple[int, int], List[int]] = defaultdict(list)  # (ix,iy) -> [cluster_ids]
        self._seen_in_cell: Set[Tuple[int, int, int]] = set()  # (ix,iy,id) de-dupe
        self.boxes: Dict[int, Tuple[float, float, float, float]] = {}  # id -> (l,t,r,b)
        self.areas: Dict[int, float] = {}

    def _ix_range(self, l: float, r: float) -> range:
        ix0 = int(l / self.bin_w)
        ix1 = int((max(r - 1e-9, l)) / self.bin_w)
        ix0 = max(0, min(ix0, self.nx - 1))
        ix1 = max(0, min(ix1, self.nx - 1))
        return range(ix0, ix1 + 1)

    def _iy_range(self, t: float, b: float) -> range:
        iy0 = int(t / self.bin_h)
        iy1 = int((max(b - 1e-9, t)) / self.bin_h)
        iy0 = max(0, min(iy0, self.ny - 1))
        iy1 = max(0, min(iy1, self.ny - 1))
        return range(iy0, iy1 + 1)

    def insert(self, cid: int, l: float, t: float, r: float, b: float):
        self.boxes[cid] = (l, t, r, b)
        self.areas[cid] = max(0.0, (r - l)) * max(0.0, (b - t))
        if r <= l or b <= t:
            return
        for ix in self._ix_range(l, r):
            for iy in self._iy_range(t, b):
                key = (ix, iy)
                # Avoid duplicates when a cluster straddles many cells
                sig = (ix, iy, cid)
                if sig not in self._seen_in_cell:
                    self.cells[key].append(cid)
                    self._seen_in_cell.add(sig)

    def candidates(self, l: float, t: float, r: float, b: float) -> Set[int]:
        if r <= l or b <= t:
            return set()
        cands = set()
        for ix in self._ix_range(l, r):
            for iy in self._iy_range(t, b):
                cands.update(self.cells.get((ix, iy), ()))
        return cands


class LayoutPostprocessor:
    """Postprocesses layout predictions by cleaning up clusters and mapping cells."""

    # Cluster type-specific parameters for overlap resolution
    OVERLAP_PARAMS = {
        "regular": {"area_threshold": 1.3, "conf_threshold": 0.05},
        "picture": {"area_threshold": 2.0, "conf_threshold": 0.3},
        "wrapper": {"area_threshold": 2.0, "conf_threshold": 0.2},
    }

    WRAPPER_TYPES = {
        DocItemLabel.FORM,
        DocItemLabel.KEY_VALUE_REGION,
        DocItemLabel.TABLE,
        DocItemLabel.DOCUMENT_INDEX,
    }
    SPECIAL_TYPES = WRAPPER_TYPES.union({DocItemLabel.PICTURE})

    CONFIDENCE_THRESHOLDS = {
        DocItemLabel.CAPTION: 0.5,
        DocItemLabel.FOOTNOTE: 0.5,
        DocItemLabel.FORMULA: 0.5,
        DocItemLabel.LIST_ITEM: 0.5,
        DocItemLabel.PAGE_FOOTER: 0.5,
        DocItemLabel.PAGE_HEADER: 0.5,
        DocItemLabel.PICTURE: 0.5,
        DocItemLabel.SECTION_HEADER: 0.45,
        DocItemLabel.TABLE: 0.5,
        DocItemLabel.TEXT: 0.5,  # 0.45,
        DocItemLabel.TITLE: 0.45,
        DocItemLabel.CODE: 0.45,
        DocItemLabel.CHECKBOX_SELECTED: 0.45,
        DocItemLabel.CHECKBOX_UNSELECTED: 0.45,
        DocItemLabel.FORM: 0.45,
        DocItemLabel.KEY_VALUE_REGION: 0.45,
        DocItemLabel.DOCUMENT_INDEX: 0.45,
    }

    LABEL_REMAPPING = {
        # DocItemLabel.DOCUMENT_INDEX: DocItemLabel.TABLE,
        DocItemLabel.TITLE: DocItemLabel.SECTION_HEADER,
    }

    def __init__(
            self, page: Page, clusters: List[Cluster], options: LayoutOptions, shared_timer=None
    ) -> None:
        """Initialize processor with page and clusters."""

        self.cells = page.cells
        self.page = page
        self.page_size = page.size
        self.all_clusters = clusters
        self.options = options
        self.regular_clusters = [
            c for c in clusters if c.label not in self.SPECIAL_TYPES
        ]
        self.special_clusters = [c for c in clusters if c.label in self.SPECIAL_TYPES]

        # Cache compatibility mode settings to avoid repeated env lookups
        self.compat_mode = os.getenv("DOCLING_GPU_COMPAT_MODE", "").lower() in ("1", "true", "yes")
        self.epsilon = 1e-4 if self.compat_mode else 0.0
        
        # Use shared timer if provided, otherwise create own
        self.timer = shared_timer if shared_timer is not None else _CPUTimer()
        
        # Track assigned cells to avoid full scan later
        self._assigned_cell_indices = None
        self._valid_cells = None
        
        # Cache cell bboxes once per page and reuse everywhere  
        self._cell_bbox = {}
        for cell in self.cells:
            txt = getattr(cell, "text", None)
            if not txt or not txt.strip():
                continue
            bb = cell.rect.to_bounding_box()
            if bb.area() > 0:
                self._cell_bbox[cell.index] = bb

        # Build spatial indices once (skip intervals for regular clusters)
        self.regular_index = SpatialClusterIndex(self.regular_clusters, build_intervals=False)
        self.picture_index = SpatialClusterIndex(
            [c for c in self.special_clusters if c.label == DocItemLabel.PICTURE], build_intervals=True
        )
        self.wrapper_index = SpatialClusterIndex(
            [c for c in self.special_clusters if c.label in self.WRAPPER_TYPES], build_intervals=True
        )

    def postprocess(self) -> Tuple[List[Cluster], List[TextCell]]:
        """Main processing pipeline with comprehensive timing."""
        
        with self.timer.time_section("postprocess_total"):
            with self.timer.time_section("process_regular"):
                self.regular_clusters = self._process_regular_clusters()
            
            with self.timer.time_section("process_special"):
                self.special_clusters = self._process_special_clusters()

            # Remove regular clusters that are included in wrappers
            with self.timer.time_section("filter_contained"):
                contained_ids = {
                    child.id
                    for wrapper in self.special_clusters
                    if wrapper.label in self.SPECIAL_TYPES
                    for child in wrapper.children
                }
                self.regular_clusters = [
                    c for c in self.regular_clusters if c.id not in contained_ids
                ]

            # Combine and sort final clusters
            with self.timer.time_section("sort_final"):
                final_clusters = self._sort_clusters(
                    self.regular_clusters + self.special_clusters, mode="id"
                )
                for cluster in final_clusters:
                    cluster.cells = self._sort_cells(cluster.cells)
                    # Also sort cells in children if any
                    for child in cluster.children:
                        child.cells = self._sort_cells(child.cells)

            with self.timer.time_section("finalize_page"):
                assert self.page.parsed_page is not None
                self.page.parsed_page.textline_cells = self.cells
                self.page.parsed_page.has_lines = len(self.cells) > 0

        return final_clusters, self.cells

    def _process_regular_clusters(self) -> List[Cluster]:
        """Process regular clusters with iterative refinement and detailed timing."""
        with self.timer.time_section("filter_regular"):
            clusters = [
                c
                for c in self.regular_clusters
                if c.confidence >= self.CONFIDENCE_THRESHOLDS[c.label]
            ]

            # Apply label remapping
            for cluster in clusters:
                if cluster.label in self.LABEL_REMAPPING:
                    cluster.label = self.LABEL_REMAPPING[cluster.label]

        # Initial cell assignment
        with self.timer.time_section("assign_cells"):
            clusters = self._assign_cells_to_clusters(clusters)

        with self.timer.time_section("filter_empty"):
            # Remove clusters with no cells (if keep_empty_clusters is False),
            # but always keep clusters with label DocItemLabel.FORMULA
            if not self.options.keep_empty_clusters:
                clusters = [
                    cluster
                    for cluster in clusters
                    if cluster.cells or cluster.label == DocItemLabel.FORMULA
                ]

        # Handle orphaned cells (only if we're creating orphan clusters)
        if self.options.create_orphan_clusters:
            with self.timer.time_section("find_unassigned"):
                unassigned = self._find_unassigned_cells(clusters)
                if unassigned:
                    next_id = max((c.id for c in self.all_clusters), default=0) + 1
                    orphan_clusters = []
                    for i, cell in enumerate(unassigned):
                        conf = cell.confidence

                        orphan_clusters.append(
                            Cluster(
                                id=next_id + i,
                                label=DocItemLabel.TEXT,
                                bbox=cell.to_bounding_box(),
                                confidence=conf,
                                cells=[cell],
                            )
                        )
                    clusters.extend(orphan_clusters)

        # Iterative refinement
        with self.timer.time_section("iterative_refinement"):
            prev_count = len(clusters) + 1
            for iteration in range(3):  # Maximum 3 iterations
                if prev_count == len(clusters):
                    break
                prev_count = len(clusters)
                
                with self.timer.time_section(f"adjust_bboxes_iter{iteration}"):
                    clusters, moved = self._adjust_cluster_bboxes(clusters)
                
                with self.timer.time_section(f"overlaps_regular_iter{iteration}"):
                    clusters, merged = self._remove_overlapping_clusters(clusters, "regular")
                
                # Early break if nothing changed
                if not moved and not merged:
                    break

        return clusters

    def _process_special_clusters(self) -> List[Cluster]:
        with self.timer.time_section("filter_special"):
            special_clusters = [
                c
                for c in self.special_clusters
                if c.confidence >= self.CONFIDENCE_THRESHOLDS[c.label]
            ]

        with self.timer.time_section("cross_type_overlaps"):
            special_clusters = self._handle_cross_type_overlaps(special_clusters)

        with self.timer.time_section("filter_full_page_pictures"):
            # Calculate page area from known page size
            assert self.page_size is not None
            page_area = self.page_size.width * self.page_size.height
            if page_area > 0:
                # Filter out full-page pictures
                special_clusters = [
                    cluster
                    for cluster in special_clusters
                    if not (
                            cluster.label == DocItemLabel.PICTURE
                            and cluster.bbox.area() / page_area > 0.90
                    )
                ]

        with self.timer.time_section("assign_children"):
            for special in special_clusters:
                contained = []
                for cluster in self.regular_clusters:
                    containment = cluster.bbox.intersection_over_self(special.bbox)
                    if containment > 0.8:
                        contained.append(cluster)

                if contained:
                    # Sort contained clusters by minimum cell ID:
                    contained = self._sort_clusters(contained, mode="id")
                    special.children = contained

                    # Adjust bbox only for Form and Key-Value-Region, not Table or Picture
                    if special.label in [DocItemLabel.FORM, DocItemLabel.KEY_VALUE_REGION]:
                        special.bbox = BoundingBox(
                            l=min(c.bbox.l for c in contained),
                            t=min(c.bbox.t for c in contained),
                            r=max(c.bbox.r for c in contained),
                            b=max(c.bbox.b for c in contained),
                        )

                    # Collect all cells from children
                    all_cells = []
                    for child in contained:
                        all_cells.extend(child.cells)
                    special.cells = self._deduplicate_cells(all_cells)
                    special.cells = self._sort_cells(special.cells)

        with self.timer.time_section("overlaps_picture"):
            picture_clusters = [
                c for c in special_clusters if c.label == DocItemLabel.PICTURE
            ]
            picture_clusters, _ = self._remove_overlapping_clusters(
                picture_clusters, "picture"
            )

        with self.timer.time_section("overlaps_wrapper"):
            wrapper_clusters = [
                c for c in special_clusters if c.label in self.WRAPPER_TYPES
            ]
            wrapper_clusters, _ = self._remove_overlapping_clusters(
                wrapper_clusters, "wrapper"
            )

        return picture_clusters + wrapper_clusters

    def _handle_cross_type_overlaps(self, special_clusters) -> List[Cluster]:
        """Handle overlaps between regular and wrapper clusters before child assignment.

        In particular, KEY_VALUE_REGION proposals that are almost identical to a TABLE
        should be removed.
        """
        wrappers_to_remove = set()

        for wrapper in special_clusters:
            if wrapper.label not in self.WRAPPER_TYPES:
                continue  # only treat KEY_VALUE_REGION for now.

            for regular in self.regular_clusters:
                if regular.label == DocItemLabel.TABLE:
                    # Calculate overlap
                    overlap_ratio = wrapper.bbox.intersection_over_self(regular.bbox)

                    conf_diff = wrapper.confidence - regular.confidence

                    # If wrapper is mostly overlapping with a TABLE, remove the wrapper
                    if (
                            overlap_ratio > 0.9 and conf_diff < 0.1
                    ):  # self.OVERLAP_PARAMS["wrapper"]["conf_threshold"]):  # 80% overlap threshold
                        wrappers_to_remove.add(wrapper.id)
                        break

        # Filter out the identified wrappers
        special_clusters = [
            cluster
            for cluster in special_clusters
            if cluster.id not in wrappers_to_remove
        ]

        return special_clusters

    def _should_prefer_cluster(
            self, candidate: Cluster, other: Cluster, params: dict
    ) -> bool:
        """Determine if candidate cluster should be preferred over other cluster based on rules.
        Returns True if candidate should be preferred, False if not."""

        # Rule 1: LIST_ITEM vs TEXT
        if (
                candidate.label == DocItemLabel.LIST_ITEM
                and other.label == DocItemLabel.TEXT
        ):
            # Check if areas are similar (within 20% of each other)
            area_ratio = candidate.bbox.area() / other.bbox.area()
            area_similarity = abs(1 - area_ratio) < 0.2
            if area_similarity:
                return True

        # Rule 2: CODE vs others
        if candidate.label == DocItemLabel.CODE:
            # Calculate how much of the other cluster is contained within the CODE cluster
            containment = other.bbox.intersection_over_self(candidate.bbox)
            if containment > 0.8:  # other is 80% contained within CODE
                return True

        # If no label-based rules matched, fall back to area/confidence thresholds
        area_ratio = candidate.bbox.area() / other.bbox.area()
        conf_diff = other.confidence - candidate.confidence

        if (
                area_ratio <= params["area_threshold"]
                and conf_diff > params["conf_threshold"]
        ):
            return False

        return True  # Default to keeping candidate if no rules triggered rejection

    def _select_best_cluster_from_group(
            self,
            group_clusters: List[Cluster],
            params: dict,
    ) -> Cluster:
        """Select best cluster from a group of overlapping clusters based on all rules."""
        current_best = None

        for candidate in group_clusters:
            should_select = True

            for other in group_clusters:
                if other == candidate:
                    continue

                if not self._should_prefer_cluster(candidate, other, params):
                    should_select = False
                    break

            if should_select:
                if current_best is None:
                    current_best = candidate
                else:
                    # If both clusters pass rules, prefer the larger one unless confidence differs significantly
                    if (
                            candidate.bbox.area() > current_best.bbox.area()
                            and current_best.confidence - candidate.confidence
                            <= params["conf_threshold"]
                    ):
                        current_best = candidate

        return current_best if current_best else group_clusters[0]

    def _remove_overlapping_clusters(
            self,
            clusters: List[Cluster],
            cluster_type: str,
            overlap_threshold: float = 0.8,
            containment_threshold: float = 0.8,
    ) -> Tuple[List[Cluster], bool]:
        if not clusters:
            return [], False

        spatial_index = (
            self.regular_index
            if cluster_type == "regular"
            else self.picture_index
            if cluster_type == "picture"
            else self.wrapper_index
        )

        # Map of currently valid clusters
        valid_clusters = {c.id: c for c in clusters}
        uf = UnionFind(valid_clusters.keys())
        params = self.OVERLAP_PARAMS[cluster_type]

        if cluster_type == "regular":
            # Small cluster count: use simple O(N²) pairwise
            if len(clusters) < 20:
                boxes = {c.id: c.bbox.as_tuple() for c in clusters}
                areas = {c.id: c.bbox.area() for c in clusters}
                ovlp_thr = overlap_threshold - self.epsilon
                cont_thr = containment_threshold - self.epsilon
                
                def overlaps(a: int, b: int) -> bool:
                    l1, t1, r1, b1 = boxes[a]
                    l2, t2, r2, b2 = boxes[b]
                    l = max(l1, l2); t = max(t1, t2); r = min(r1, r2); b = min(b1, b2)
                    iw = r - l; ih = b - t
                    if iw <= 0.0 or ih <= 0.0:
                        return False
                    interA = iw * ih
                    aa, bb = areas[a], areas[b]
                    if aa > 0.0 and interA / aa >= cont_thr: return True
                    if bb > 0.0 and interA / bb >= cont_thr: return True
                    denom = aa + bb - interA
                    return denom > 0.0 and (interA / denom) >= ovlp_thr
                
                # Simple pairwise check
                cluster_ids = [c.id for c in clusters]
                for i, cid in enumerate(cluster_ids):
                    for oid in cluster_ids[i+1:]:
                        if cid not in valid_clusters or oid not in valid_clusters:
                            continue
                        if overlaps(cid, oid):
                            uf.union(cid, oid)
            else:
                # Grid bucketing for dense pages
                boxes = {c.id: c.bbox.as_tuple() for c in clusters}
                areas = {c.id: c.bbox.area() for c in clusters}

                # Page size or extents
                if self.page_size is not None:
                    page_w, page_h = self.page_size.width, self.page_size.height
                else:
                    # Fallback to cluster extents
                    min_l = min(l for (l, _, _, _) in boxes.values())
                    min_t = min(t for (_, t, _, _) in boxes.values())
                    max_r = max(r for (_, _, r, _) in boxes.values())
                    max_b = max(b for (_, _, _, b) in boxes.values())
                    page_w = max_r - min_l
                    page_h = max_b - min_t

                # Median w/h of regular clusters
                ws = sorted((r - l) for (l, t, r, b) in boxes.values() if r > l)
                hs = sorted((b - t) for (l, t, r, b) in boxes.values() if b > t)
                med_w = ws[len(ws) // 2] if ws else max(1.0, page_w / 12.0)
                med_h = hs[len(hs) // 2] if hs else max(1.0, page_h / 24.0)

                bin_w = max(page_w / 60.0, 1.5 * med_w)  # Clamp to keep bins reasonable
                bin_h = max(page_h / 60.0, 1.5 * med_h)

                grid = GridIndex(page_w, page_h, bin_w, bin_h)
                for cid, (l, t, r, b) in boxes.items():
                    grid.insert(cid, l, t, r, b)

                ovlp_thr = overlap_threshold - self.epsilon
                cont_thr = containment_threshold - self.epsilon

                def overlaps(a: int, b: int) -> bool:
                    l1, t1, r1, b1 = boxes[a]
                    l2, t2, r2, b2 = boxes[b]
                    l = max(l1, l2); t = max(t1, t2); r = min(r1, r2); b = min(b1, b2)
                    iw = r - l; ih = b - t
                    if iw <= 0.0 or ih <= 0.0:
                        return False
                    interA = iw * ih
                    aa, bb = areas[a], areas[b]
                    # Containment shortcut
                    if aa > 0.0 and interA / aa >= cont_thr: return True
                    if bb > 0.0 and interA / bb >= cont_thr: return True
                    denom = aa + bb - interA
                    return denom > 0.0 and (interA / denom) >= ovlp_thr

                # Union only within shared grid cells
                for c in clusters:
                    cid = c.id
                    l1, t1, r1, b1 = boxes[cid]
                    # Candidate set is tiny now (sorted for deterministic order)
                    for oid in sorted(grid.candidates(l1, t1, r1, b1)):
                        if oid <= cid or oid not in valid_clusters:
                            continue
                        # Quick AABB reject by tuples
                        l2, t2, r2, b2 = boxes[oid]
                        if l2 >= r1 or r2 <= l1 or t2 >= b1 or b2 <= t1:
                            continue
                        if overlaps(cid, oid):
                            uf.union(cid, oid)
        else:
            # Keep existing logic for picture/wrapper clusters
            for cluster in clusters:
                candidates = spatial_index.find_candidates(cluster.bbox)
                candidates &= valid_clusters.keys()
                candidates.discard(cluster.id)

                for other_id in candidates:
                    if spatial_index.check_overlap(
                            cluster.bbox,
                            valid_clusters[other_id].bbox,
                            overlap_threshold,
                            containment_threshold,
                            self.epsilon,
                    ):
                        uf.union(cluster.id, other_id)

        result = []
        merged = False
        for group in uf.get_groups().values():
            if len(group) == 1:
                result.append(valid_clusters[group[0]])
                continue

            merged = True  # Something was merged
            group_clusters = [valid_clusters[cid] for cid in group]
            best = self._select_best_cluster_from_group(group_clusters, params)

            # Simple cell merging - no special cases
            for cluster in group_clusters:
                if cluster != best:
                    best.cells.extend(cluster.cells)

            best.cells = self._deduplicate_cells(best.cells)
            best.cells = self._sort_cells(best.cells)
            result.append(best)

        return result, merged

    def _select_best_cluster(
            self,
            clusters: List[Cluster],
            area_threshold: float,
            conf_threshold: float,
    ) -> Cluster:
        """Iteratively select best cluster based on area and confidence thresholds."""
        current_best = None
        for candidate in clusters:
            should_select = True
            for other in clusters:
                if other == candidate:
                    continue

                area_ratio = candidate.bbox.area() / other.bbox.area()
                conf_diff = other.confidence - candidate.confidence

                if area_ratio <= area_threshold and conf_diff > conf_threshold:
                    should_select = False
                    break

            if should_select:
                if current_best is None or (
                        candidate.bbox.area() > current_best.bbox.area()
                        and current_best.confidence - candidate.confidence <= conf_threshold
                ):
                    current_best = candidate

        return current_best if current_best else clusters[0]

    def _deduplicate_cells(self, cells: List[TextCell]) -> List[TextCell]:
        """Ensure each cell appears only once, maintaining order of first appearance."""
        if not cells:
            return []
        
        # Fast path: skip dedup when already unique (common case after assignment)
        if len(cells) <= 1:
            return cells
        
        # Quick uniqueness check by scanning indices
        indices = [cell.index for cell in cells]
        if len(set(indices)) == len(indices):
            return cells  # Already unique
        
        # Fallback: full deduplication
        seen = {}
        result = []
        for cell in cells:
            if cell.index not in seen:
                seen[cell.index] = True
                result.append(cell)
        return result

    def _assign_cells_to_clusters(
            self, clusters: List[Cluster], min_overlap: float = 0.2
    ) -> List[Cluster]:
        """Assign cells to best overlapping cluster using GridIndex optimization."""
        # Initialize clusters and track first cell index for fast sorting
        for cluster in clusters:
            cluster.cells = []
            cluster._first_cell_index = sys.maxsize

        # Early exit for empty inputs
        if not clusters or not self.cells:
            return clusters
            
        # Use cached cell bboxes and track assigned cells
        valid_cells = [c for c in self.cells if c.index in self._cell_bbox]
        self._valid_cells = valid_cells  # save for later unassigned check
        self._assigned_cell_indices = set()

        # Build geometry arrays for fast access
        id_to_cluster = {c.id: c for c in clusters}
        cluster_boxes = {c.id: c.bbox.as_tuple() for c in clusters}
        
        # Build grid for assignment (reuse page size logic from overlap)
        if self.page_size is not None:
            page_w, page_h = self.page_size.width, self.page_size.height
        else:
            # Fallback to cluster extents
            min_l = min(l for (l, _, _, _) in cluster_boxes.values())
            min_t = min(t for (_, t, _, _) in cluster_boxes.values())
            max_r = max(r for (_, _, r, _) in cluster_boxes.values())
            max_b = max(b for (_, _, _, b) in cluster_boxes.values())
            page_w = max_r - min_l
            page_h = max_b - min_t

        # Use median-based bin sizing
        ws = sorted((r - l) for (l, t, r, b) in cluster_boxes.values() if r > l)
        hs = sorted((b - t) for (l, t, r, b) in cluster_boxes.values() if b > t)
        med_w = ws[len(ws) // 2] if ws else max(1.0, page_w / 12.0)
        med_h = hs[len(hs) // 2] if hs else max(1.0, page_h / 24.0)

        bin_w = max(page_w / 60.0, 1.5 * med_w)
        bin_h = max(page_h / 60.0, 1.5 * med_h)

        # Build assignment grid
        assign_grid = GridIndex(page_w, page_h, bin_w, bin_h)
        for cid, (l, t, r, b) in cluster_boxes.items():
            assign_grid.insert(cid, l, t, r, b)
        
        # Assign each cell to best overlapping cluster
        for cell in valid_cells:
            cell_bbox = self._cell_bbox[cell.index]
            best_overlap = min_overlap
            best_cluster = None

            # Get cell bbox coords once
            lx, ty, rx, by = cell_bbox.as_tuple()

            # Query grid for candidate clusters (sorted for deterministic order)
            for cluster_id in sorted(assign_grid.candidates(lx, ty, rx, by)):
                cluster = id_to_cluster.get(cluster_id)
                if cluster is None:
                    continue
                
                # Fast AABB reject before expensive overlap computation
                l2, t2, r2, b2 = cluster_boxes[cluster_id]
                if l2 >= rx or r2 <= lx or t2 >= by or b2 <= ty:
                    continue
                    
                overlap_ratio = cell_bbox.intersection_over_self(cluster.bbox)
                if overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_cluster = cluster
                    # Fast break when we find very good overlap
                    if best_overlap >= 0.95:
                        break

            if best_cluster is not None:
                best_cluster.cells.append(cell)
                # Track first cell index for fast sorting later
                if cell.index < best_cluster._first_cell_index:
                    best_cluster._first_cell_index = cell.index
                # Track assignment to avoid full scan later
                self._assigned_cell_indices.add(cell.index)

        # Deduplicate cells in each cluster after assignment
        for cluster in clusters:
            cluster.cells = self._deduplicate_cells(cluster.cells)

        return clusters

    def _find_unassigned_cells(self, clusters: List[Cluster]) -> List[TextCell]:
        """Find cells not assigned to any cluster."""
        # Use tracked assigned indices if available (much faster)
        if self._assigned_cell_indices is not None and self._valid_cells is not None:
            return [c for c in self._valid_cells if c.index not in self._assigned_cell_indices]
        
        # Fallback to full scan (shouldn't happen in normal flow)
        assigned = {cell.index for cluster in clusters for cell in cluster.cells}
        return [
            cell
            for cell in self.cells
            if cell.index not in assigned and cell.text.strip()
        ]

    def _adjust_cluster_bboxes(self, clusters: List[Cluster]) -> Tuple[List[Cluster], bool]:
        """Adjust cluster bounding boxes to contain their cells. Returns (clusters, changed)."""
        changed = False
        for cl in clusters:
            if not cl.cells:
                continue
            
            # Single-pass min/max calculation
            first = True
            nl = nt = nr = nb = 0.0
            for cell in cl.cells:
                bb = self._cell_bbox.get(cell.index)
                if not bb:
                    continue
                if first:
                    nl, nt, nr, nb = bb.l, bb.t, bb.r, bb.b
                    first = False
                else:
                    if bb.l < nl: nl = bb.l
                    if bb.t < nt: nt = bb.t
                    if bb.r > nr: nr = bb.r
                    if bb.b > nb: nb = bb.b
            
            if first:  # No valid bb seen
                continue
                
            if cl.label == DocItemLabel.TABLE:
                # For tables, take union of current bbox and cells bbox
                if cl.bbox.l < nl: nl = cl.bbox.l
                if cl.bbox.t < nt: nt = cl.bbox.t
                if cl.bbox.r > nr: nr = cl.bbox.r
                if cl.bbox.b > nb: nb = cl.bbox.b
            
            # Compare without allocating new BoundingBox unless changed
            if (nl, nt, nr, nb) != (cl.bbox.l, cl.bbox.t, cl.bbox.r, cl.bbox.b):
                cl.bbox = BoundingBox(l=nl, t=nt, r=nr, b=nb)
                changed = True

        return clusters, changed

    def _sort_cells(self, cells: List[TextCell]) -> List[TextCell]:
        """Sort cells in native reading order."""
        return sorted(cells, key=lambda c: (c.index))

    def _sort_clusters(
            self, clusters: List[Cluster], mode: str = "id"
    ) -> List[Cluster]:
        """Sort clusters in reading order (top-to-bottom, left-to-right)."""
        if mode == "id":  # sort in the order the cells are printed in the PDF.
            return sorted(
                clusters,
                key=lambda cluster: (
                    cluster._first_cell_index,
                    cluster.bbox.t,
                    cluster.bbox.l,
                ),
            )
        elif mode == "tblr":  # Sort top-to-bottom, then left-to-right ("row first")
            return sorted(
                clusters, key=lambda cluster: (cluster.bbox.t, cluster.bbox.l)
            )
        elif mode == "lrtb":  # Sort left-to-right, then top-to-bottom ("column first")
            return sorted(
                clusters, key=lambda cluster: (cluster.bbox.l, cluster.bbox.t)
            )
        else:
            return clusters
