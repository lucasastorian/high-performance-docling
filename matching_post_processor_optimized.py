#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import logging
import math

import numpy as np
import docling_ibm_models.tableformer.settings as s

from tf_cell_matcher import CellMatcher

LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG


class MatchingPostProcessor:
    r"""
    The MatchingPostProcessor aims to improve the matchings between the predicted table cells and
    the pdf cells
    """

    def __init__(self, config):
        self._config = config
        self._cell_matcher = CellMatcher(config)

    def _log(self):
        # Setup a custom logger
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    def _get_table_dimension(self, table_cells):
        r"""
        Get dimensions (columns, rows) of a table from table_cells

        Parameters
        ----------
        table_cells : list of dict
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"

        Returns
        -------
        columns : integer,
        rows : integer,
        max_cell_id : integer,
            highest cell_id in table_cells
        """
        # Vectorized version - O(n) single pass
        if not table_cells:
            return 1, 1, 0
            
        cols = np.fromiter((c["column_id"] for c in table_cells), int)
        rows = np.fromiter((c["row_id"] for c in table_cells), int)
        ids = np.fromiter((c["cell_id"] for c in table_cells), int)
        
        return cols.max() + 1, rows.max() + 1, ids.max()

    def _get_good_bad_cells_in_column(self, table_cells, column, matched_mask, by_col, cell_classes):
        r"""
        1. step - FULLY OPTIMIZED
        Get good/bad IOU predicted cells for each structural column using precomputed structures.
        O(cells_in_column) with true vectorization (no hidden Python loops).

        Parameters
        ----------
        table_cells : list of dict
        column : int
        matched_mask : np.array - boolean mask for matched cell IDs
        by_col : list of lists - precomputed column indices
        cell_classes : np.array - precomputed cell classes

        Returns
        -------
        good_table_cells, bad_table_cells : list of dict
        """
        # Get indices for this column
        idxs = by_col[column] if column < len(by_col) else []
        if not idxs:
            return [], []
            
        # Vectorized operations on indices
        ids = np.fromiter((table_cells[i]["cell_id"] for i in idxs), dtype=int)
        cls = cell_classes[idxs]
        
        # True vectorization - O(k) pure NumPy indexing
        allow = cls > 1
        matched = matched_mask[ids]  # No Python loop!
        good_mask = allow & matched
        
        # Build result lists
        good = [table_cells[i] for i, g in zip(idxs, good_mask) if g]
        bad = [table_cells[i] for i, g in zip(idxs, good_mask) if not g]
        
        return good, bad

    def _delete_column_from_table(self, table_cells, column):
        r"""
        1.a. step
        If all IOU in a column are bad - eliminate column (from bboxes and structure)

        Parameters
        ----------
        table_cells : list of dict
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"
        column : integer
            Index of a column

        Returns
        -------
        new_table_cells : list of dict
        """
        new_table_cells = []

        for cell in table_cells:
            if cell["column_id"] < column:
                new_table_cells.append(cell)
            if cell["column_id"] > column:
                new_cell = {
                    "bbox": cell["bbox"],
                    "cell_id": cell["cell_id"],
                    "column_id": cell["column_id"] - 1,
                    "label": cell["label"],
                    "row_id": cell["row_id"],
                    "cell_class": cell["cell_class"],
                }
                new_table_cells.append(new_cell)

        return new_table_cells

    def _find_alignment_in_column(self, cells):
        r"""
        2. step
        Find alignment of good IOU cells per column

        Parameters
        ----------
        cells : list of dict
            Cells in a column
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"

        Returns
        -------
        alignment : string
            column general alignment can be: "left", "right", "center"
        """
        possible_alignments = ["left", "middle", "right"]
        alignment = "left"  # left / right / center

        lefts = []
        rights = []
        middles = []

        for cell in cells:
            x_left = cell["bbox"][0]
            x_right = cell["bbox"][2]
            x_middle = (x_left + x_right) / 2
            lefts.append(x_left)
            rights.append(x_right)
            middles.append(x_middle)

        if len(lefts) > 0:
            delta_left = max(lefts) - min(lefts)
            delta_middle = max(middles) - min(middles)
            delta_right = max(rights) - min(rights)

            deltas = [delta_left, delta_middle, delta_right]
            align_index = deltas.index(min(deltas))
            alignment = possible_alignments[align_index]

        return alignment

    def _get_median_pos_size(self, cells, alignment):
        r"""
        3. step
        Get median* (according to alignment) "bbox left/middle/right X" coord
        for good IOU cells, get median* cell size in a column.

        Parameters
        ----------
        cells : list of dict
            Cells in a column
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"
        alignment : string
            column general alignment can be: "left", "right", "center"

        Returns
        -------
        median_x : number
            Median X position of a cell (according to alignment)
        median_y : number
            Median Y position of a cell (according to alignment)
        median_width : number
            Median width of a cell
        median_height : number
            Median height of a cell
        """
        median_x = 0
        median_y = 0
        median_width = 1
        median_height = 1

        coords_x = []
        coords_y = []
        widths = []
        heights = []

        for cell in cells:
            if "rowspan_val" not in cell:
                if "colspan_val" not in cell:
                    if cell["cell_class"] > 1:
                        # Use left alignment
                        x_coord = cell["bbox"][0]
                        if alignment == "middle":
                            # Use middle alignment
                            x_coord = (cell["bbox"][2] + cell["bbox"][0]) / 2
                        if alignment == "right":
                            # Use right alignment
                            x_coord = cell["bbox"][2]

                        coords_x.append(x_coord)
                        y_coord = cell["bbox"][1]
                        coords_y.append(y_coord)

                        width = cell["bbox"][2] - cell["bbox"][0]
                        widths.append(width)
                        height = cell["bbox"][3] - cell["bbox"][1]
                        heights.append(height)
                    else:
                        self._log().debug("Empty cells not considered in medians")
                        self._log().debug(cell)
                else:
                    self._log().debug("Colspans not considered in medians")
                    self._log().debug(cell)
            else:
                self._log().debug("Rowspans not considered in medians")
                self._log().debug(cell)

        if len(coords_x) > 0:
            median_x = float(np.median(coords_x))
        if len(coords_y) > 0:
            median_y = float(np.median(coords_y))
        if len(widths) > 0:
            median_width = float(np.median(widths))
        if len(heights) > 0:
            median_height = float(np.median(heights))
        return median_x, median_y, median_width, median_height

    def _move_cells_to_left_pos(
        self, cells, median_x, rescale, median_width, median_height, alignment
    ):
        r"""
        4. step
        Move bad cells to the median* (left/middle/right) good in a column
        (Additionally), re-size cell to median* size of cells in a column

        Parameters
        ----------
        cells : list of dict
            Cells in a column
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"
        median_x : number
            Median X position of a cell (according to alignment)
        rescale : boolean
            should cells be re-sized to median or not
        median_width : number
            Median width of a cell
        median_height : number
            Median height of a cell
        alignment : string
            column general alignment can be: "left", "right", "center"

        Returns
        -------

        new_table_cells : list of dict
            Cells in a column
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"
        """
        if not cells:
            return []

        # Vectorized bbox computation
        b = np.asarray([c["bbox"] for c in cells], dtype=np.float32)
        x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        widths = x2 - x1

        # Determine target width based on rescale
        if rescale:
            target_w = float(median_width)
            new_y2 = y1 + float(median_height)
        else:
            target_w = widths
            new_y2 = y2

        # Default alignment is "left"
        new_x1 = np.full_like(widths, float(median_x))
        new_x2 = new_x1 + target_w
        new_y1 = y1

        # Adjust for "middle" alignment
        if alignment == "middle":
            new_x1 = float(median_x) - (target_w / 2.0)
            new_x2 = new_x1 + target_w

        # Adjust for "right" alignment
        elif alignment == "right":
            new_x1 = float(median_x) - target_w
            new_x2 = np.full_like(widths, float(median_x))

        # Stack into new bbox array
        new_b = np.stack([new_x1, new_y1, new_x2, new_y2], axis=1)

        # Rebuild dicts with updated bbox (single pass)
        new_table_cells = []
        for i, cell in enumerate(cells):
            new_cell = {
                "bbox": new_b[i].tolist(),
                "cell_id": cell["cell_id"],
                "column_id": cell["column_id"],
                "label": cell.get("label", ""),
                "row_id": cell["row_id"],
                "cell_class": cell.get("cell_class", 0),
            }
            # Preserve optional spans if present
            if "rowspan_val" in cell:
                new_cell["rowspan_val"] = cell["rowspan_val"]
            if "colspan_val" in cell:
                new_cell["colspan_val"] = cell["colspan_val"]
            new_table_cells.append(new_cell)
        
        return new_table_cells

    def _run_intersection_match(self, cell_matcher, table_cells, pdf_cells):
        r"""
        5. step
        Generate new matches, run Intersection over cell(pdf) on a table cells

        Parameters
        ----------
        cell_matcher : CellMatcher
            src.data_management.cell_matcher
        table_cells : list of dict
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"
        pdf_cells : list of dict
            List of PDF cells as defined by Docling

        Returns
        -------
        clean_matches : dictionary of lists of table_cells
            A dictionary which is indexed by the pdf_cell_id (int) as key and the value is a list
            of the table_cells that fall inside that pdf cell
        """
        new_matches, matches_counter = cell_matcher._intersection_over_pdf_match(
            table_cells, pdf_cells
        )
        # Keep keys as int - no string conversion
        return new_matches

    def _find_overlapping(self, table_cells):
        """
        OPTIMIZED: Use grid bucketing for O(n) overlap detection instead of O(n²).
        """
        if len(table_cells) < 64:  # Small tables: use simple approach
            return self._find_overlapping_simple(table_cells)
            
        # Grid-based approach for larger tables
        b = np.array([c["bbox"] for c in table_cells], dtype=np.float32)
        if len(b) == 0:
            return table_cells
            
        # Compute grid size based on median cell dimensions
        w = np.median(b[:, 2] - b[:, 0])
        h = np.median(b[:, 3] - b[:, 1])
        gw = max(w, 1.0)
        gh = max(h, 1.0)
        
        # Bucket cells by grid coordinates
        buckets = {}
        for i, (x1, y1, x2, y2) in enumerate(b):
            ix1, ix2 = int(x1 // gw), int(x2 // gw)
            iy1, iy2 = int(y1 // gh), int(y2 // gh)
            # Add cell to all buckets it overlaps
            for ix in range(ix1, ix2 + 1):
                for iy in range(iy1, iy2 + 1):
                    buckets.setdefault((ix, iy), []).append(i)
        
        def overlaps(i, j):
            a, b = table_cells[i]["bbox"], table_cells[j]["bbox"]
            return not (a[0] >= b[2] or a[2] <= b[0] or a[1] >= b[3] or a[3] <= b[1])
        
        # Only check pairs within same/adjacent buckets
        seen = set()
        for bucket_idxs in buckets.values():
            for i in range(len(bucket_idxs)):
                for j in range(i + 1, len(bucket_idxs)):
                    idx_i, idx_j = bucket_idxs[i], bucket_idxs[j]
                    pair = (min(idx_i, idx_j), max(idx_i, idx_j))
                    if pair in seen:
                        continue
                    seen.add(pair)
                    if overlaps(idx_i, idx_j):
                        table_cells[idx_i], table_cells[idx_j] = self._correct_overlap(
                            table_cells[idx_i], table_cells[idx_j]
                        )
        
        return table_cells
    
    def _find_overlapping_simple(self, table_cells):
        """Simple O(n²) overlap detection for small tables."""
        def correct_overlap(box1, box2):
            # Extract coordinates from the bounding boxes
            x1_min, y1_min, x1_max, y1_max = box1["bbox"]
            x2_min, y2_min, x2_max, y2_max = box2["bbox"]

            # Calculate the overlap in both x and y directions
            overlap_x = min(x1_max, x2_max) - max(x1_min, x2_min)
            overlap_y = min(y1_max, y2_max) - max(y1_min, y2_min)

            # If there is no overlap, return the original boxes
            if overlap_x <= 0 or overlap_y <= 0:
                return box1, box2

            # Decide how to push the boxes apart
            if overlap_x < overlap_y:
                # Push horizontally
                if x1_min < x2_min:
                    # Move box1 to the left and box2 to the right
                    box1["bbox"][2] -= math.ceil(overlap_x / 2) + 2
                    box2["bbox"][0] += math.floor(overlap_x / 2)
                else:
                    # Move box2 to the left and box1 to the right
                    box2["bbox"][2] -= math.ceil(overlap_x / 2) + 2
                    box1["bbox"][0] += math.floor(overlap_x / 2)
            else:
                # Push vertically
                if y1_min < y2_min:
                    # Move box1 up and box2 down
                    box1["bbox"][3] -= math.ceil(overlap_y / 2) + 2
                    box2["bbox"][1] += math.floor(overlap_y / 2)
                else:
                    # Move box2 up and box1 down
                    box2["bbox"][3] -= math.ceil(overlap_y / 2) + 2
                    box1["bbox"][1] += math.floor(overlap_y / 2)

            # Will flip coordinates in proper order, if previous operations reversed it
            box1["bbox"] = [
                min(box1["bbox"][0], box1["bbox"][2]),
                min(box1["bbox"][1], box1["bbox"][3]),
                max(box1["bbox"][0], box1["bbox"][2]),
                max(box1["bbox"][1], box1["bbox"][3]),
            ]
            box2["bbox"] = [
                min(box2["bbox"][0], box2["bbox"][2]),
                min(box2["bbox"][1], box2["bbox"][3]),
                max(box2["bbox"][0], box2["bbox"][2]),
                max(box2["bbox"][1], box2["bbox"][3]),
            ]

            return box1, box2
        
        def do_boxes_overlap(box1, box2):
            B1 = box1["bbox"]
            B2 = box2["bbox"]
            if (
                (B1[0] >= B2[2])
                or (B1[2] <= B2[0])
                or (B1[3] <= B2[1])
                or (B1[1] >= B2[3])
            ):
                return False
            else:
                return True

        def find_overlapping_pairs_indexes(bboxes):
            overlapping_indexes = []
            # Compare each box with every other box (combinations)
            for i in range(len(bboxes)):
                for j in range(i + 1, len(bboxes)):
                    if i != j:
                        if bboxes[i] != bboxes[j]:
                            if do_boxes_overlap(bboxes[i], bboxes[j]):
                                bboxes[i], bboxes[j] = self._correct_overlap(
                                    bboxes[i], bboxes[j]
                                )

            return overlapping_indexes, bboxes

        overlapping_indexes, table_cells = find_overlapping_pairs_indexes(table_cells)
        return table_cells
    
    def _correct_overlap(self, box1, box2):
        """Extract the overlap correction logic for reuse."""
        # Extract coordinates from the bounding boxes
        x1_min, y1_min, x1_max, y1_max = box1["bbox"]
        x2_min, y2_min, x2_max, y2_max = box2["bbox"]

        # Calculate the overlap in both x and y directions
        overlap_x = min(x1_max, x2_max) - max(x1_min, x2_min)
        overlap_y = min(y1_max, y2_max) - max(y1_min, y2_min)

        # If there is no overlap, return the original boxes
        if overlap_x <= 0 or overlap_y <= 0:
            return box1, box2

        # Decide how to push the boxes apart
        if overlap_x < overlap_y:
            # Push horizontally
            if x1_min < x2_min:
                # Move box1 to the left and box2 to the right
                box1["bbox"][2] -= math.ceil(overlap_x / 2) + 2
                box2["bbox"][0] += math.floor(overlap_x / 2)
            else:
                # Move box2 to the left and box1 to the right
                box2["bbox"][2] -= math.ceil(overlap_x / 2) + 2
                box1["bbox"][0] += math.floor(overlap_x / 2)
        else:
            # Push vertically
            if y1_min < y2_min:
                # Move box1 up and box2 down
                box1["bbox"][3] -= math.ceil(overlap_y / 2) + 2
                box2["bbox"][1] += math.floor(overlap_y / 2)
            else:
                # Move box2 up and box1 down
                box2["bbox"][3] -= math.ceil(overlap_y / 2) + 2
                box1["bbox"][1] += math.floor(overlap_y / 2)

        # Fix coordinates if they got flipped
        for box in [box1, box2]:
            box["bbox"] = [
                min(box["bbox"][0], box["bbox"][2]),
                min(box["bbox"][1], box["bbox"][3]),
                max(box["bbox"][0], box["bbox"][2]),
                max(box["bbox"][1], box["bbox"][3]),
            ]

        return box1, box2

    def _align_table_cells_to_pdf(self, table_cells, pdf_cells, matches):
        """
        Align table cell bboxes with good matches to encapsulate matching pdf cells.
        FULLY OPTIMIZED: True single-pass aggregation with no intermediate lists.
        """
        if not matches:
            return table_cells
            
        # Build lookups
        pdf_bbox = {p["id"]: p["bbox"] for p in pdf_cells}
        cells = {c["cell_id"]: c for c in table_cells}
        
        # Single-pass aggregation: cell_id -> [minx, miny, maxx, maxy]
        acc = {}
        for pid, mlist in matches.items():
            bb = pdf_bbox.get(pid)
            if bb is None:
                continue
                
            seen = set()  # Avoid processing same table cell multiple times per PDF cell
            for m in mlist:
                cid = m["table_cell_id"]
                if cid in seen or cid not in cells:
                    continue
                seen.add(cid)
                
                # Update accumulated bbox
                if cid not in acc:
                    acc[cid] = [bb[0], bb[1], bb[2], bb[3]]
                else:
                    a = acc[cid]
                    a[0] = min(a[0], bb[0])  # min x1
                    a[1] = min(a[1], bb[1])  # min y1
                    a[2] = max(a[2], bb[2])  # max x2
                    a[3] = max(a[3], bb[3])  # max y2
        
        # Build output in single pass
        out = []
        for c in table_cells:
            if c["cell_id"] in acc:
                nc = dict(c)
                nc["bbox"] = acc[c["cell_id"]]
                nc.setdefault("cell_class", 2)
                out.append(nc)
            else:
                out.append(c)
        
        return out

    def _deduplicate_cells(self, tab_columns, table_cells, iou_matches, ioc_matches):
        """
        De-duplicate structural columns using a single pass over matches (O(M)),
        global duplicate detection, and proper column reindexing.
        Returns: new_table_cells, new_matches (IOC-based), new_tab_columns
        """
        # 0) Maps
        cell2col = {c["cell_id"]: c["column_id"] for c in table_cells}

        # 1) Accumulate per-column scores and pdf hit-sets in ONE pass
        pdf_sets = [set() for _ in range(tab_columns)]
        col_score = [0.0] * tab_columns

        def acc(matches, is_iou):
            for pdf_id, mlist in matches.items():
                for m in mlist:
                    cid = m["table_cell_id"]
                    col = cell2col.get(cid)
                    if col is None or col >= tab_columns:
                        continue
                    if is_iou:
                        s = m.get("iou", m.get("iopdf", 0.0))
                    else:
                        s = m.get("iopdf", 0.0)
                    col_score[col] += float(s)
                    pdf_sets[col].add(pdf_id)

        acc(iou_matches, True)
        acc(ioc_matches, False)

        # 2) Group identical pdf-sets first (cheap win)
        from collections import defaultdict
        group = defaultdict(list)
        for col, s in enumerate(pdf_sets):
            group[frozenset(s)].append(col)

        to_eliminate = set()
        for cols in group.values():
            if len(cols) > 1:
                # keep the highest score; kill the rest
                keep = max(cols, key=lambda c: col_score[c])
                for c in cols:
                    if c != keep:
                        to_eliminate.add(c)

        # 3) Compare near-duplicates across different sets (global)
        # Use Jaccard; fall back to containment if you prefer.
        def jacc(a, b):
            if not a and not b:  # both empty -> identical
                return 1.0
            inter = len(a & b)
            union = len(a | b)
            return inter / union if union else 0.0

        THRESH = self._config.get("dup_col_jaccard", 0.6)
        C = tab_columns
        for i in range(C):
            if i in to_eliminate:
                continue
            for j in range(i + 1, C):
                if j in to_eliminate:
                    continue
                sim = jacc(pdf_sets[i], pdf_sets[j])
                if sim >= THRESH:
                    # eliminate the lower-score column; tie → keep lower index
                    if (col_score[i], -i) >= (col_score[j], -j):
                        to_eliminate.add(j)
                    else:
                        to_eliminate.add(i)
                        break  # i is gone; stop comparing it

        # 4) Build remap for surviving columns → 0..K-1
        survivors = [c for c in range(C) if c not in to_eliminate]
        col_remap = {old: new for new, old in enumerate(survivors)}

        # 5) Filter table cells and reindex column_id
        removed_cell_ids = set()
        for oldcol in to_eliminate:
            removed_cell_ids.update([c["cell_id"] for c in table_cells if c["column_id"] == oldcol])

        new_table_cells = []
        for c in table_cells:
            oldcol = c["column_id"]
            if oldcol in to_eliminate:
                continue
            nc = dict(c)
            nc["column_id"] = col_remap[oldcol]
            new_table_cells.append(nc)

        # 6) Filter IOC matches (these feed your final assignment)
        new_matches = {}
        for pdf_id, mlist in ioc_matches.items():
            fl = []
            for m in mlist:
                cid = m["table_cell_id"]
                if cid in removed_cell_ids:
                    continue
                fl.append(m)
            if fl:
                new_matches[pdf_id] = fl

        new_tab_columns = len(survivors)
        return new_table_cells, new_matches, new_tab_columns

    def _do_final_asignment(self, table_cells, iou_matches, ioc_matches):
        r"""
        8. step

        Do final assignment of table bbox to pdf cell based on saved scores,
        either preferring IOU over PDF Intersection, and higher Intersection over lower,
        or just use PDF Intersection
        Rule: 1 Table cell can contain many PDF cells,
            but each PDF cell has to be asigned to one Table cell
        Rule: Do not discard table bboxes at this point, asign all of them

        Iterate over matches, if PDF cell has more than 1 table cell match:
        Go over all other matches and delete tab_cell match of lower score
        (prefer iou match over ioc match)

        Parameters
        ----------
        table_cells : list of dict
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id", "bbox", "label"
        iou_matches : dictionary of lists of table_cells
            Cell matches done using Intersection Over Union (IOU) method
        ioc_matches : dictionary of lists of table_cells
            Cell matches done using Intersection Over (PDF) Cell method

        Returns
        -------
        new_matches : dictionary of lists of table_cells
            New matches with final table cell asignments
        """
        new_matches = {}

        for pdf_cell_id, pdf_cell_matches in ioc_matches.items():
            max_ioc_match = max(pdf_cell_matches, key=lambda x: x["iopdf"])
            new_matches[pdf_cell_id] = [max_ioc_match]

        return new_matches

    def _merge_two_bboxes(self, bbox1, bbox2):
        r"""
        Merge two bboxes into one bboxes that encompasses the two

        Parameters
        ----------
        bbox1 : list of numbers
            bbox to be merged described as two corners [x1, y1, x2, y2]
        bbox1 : list of numbers
            bbox to be merged described as two corners [x1, y1, x2, y2]

        Returns
        -------
        bbox_result : list of numbers
            bbox that encompasses two input bboxes
        """
        bbox_result = [-1, -1, -1, -1]
        bbox_result[0] = min([bbox1[0], bbox2[0]])
        bbox_result[1] = min([bbox1[1], bbox2[1]])
        bbox_result[2] = max([bbox1[2], bbox2[2]])
        bbox_result[3] = max([bbox1[3], bbox2[3]])
        return bbox_result

    def _pick_orphan_cells_vectorized(
        self, tab_rows, tab_cols, max_cell_id, table_cells, pdf_cells, matches, by_col, by_row, cell_classes
    ):
        """
        Optimized orphan cell detection using NumPy vectorization and precomputed structures.
        """
        new_matches = matches.copy()
        new_table_cells = table_cells.copy()
        
        # Skip if no pdf_cells
        if not pdf_cells:
            return new_matches, new_table_cells, max_cell_id
            
        # Convert to numpy for vectorized operations
        pdf_ids = np.array([p["id"] for p in pdf_cells], dtype=np.int64)
        pdf_bboxes = np.array([p["bbox"] for p in pdf_cells], dtype=np.float32)
        
        # Find orphan cells (not in matches)
        matched_ids = set(matches.keys())
        orphan_mask = np.array([pid not in matched_ids for pid in pdf_ids])
        
        if not orphan_mask.any():
            return new_matches, new_table_cells, max_cell_id
            
        orphan_ids = pdf_ids[orphan_mask]
        orphan_bboxes = pdf_bboxes[orphan_mask]
        
        # Process rows efficiently using passed precomputed structures
        row_assignments = {}
        for row_idx in range(tab_rows):
            # Get row cells efficiently using index
            row_indices = by_row[row_idx] if row_idx < len(by_row) else []
            row_cells = [table_cells[i] for i in row_indices 
                        if "rowspan_val" not in table_cells[i] and cell_classes[i] > 1]
            
            if not row_cells:
                continue
                
            # Get row band
            row_y1 = min(c["bbox"][1] for c in row_cells)
            row_y2 = max(c["bbox"][3] for c in row_cells)
            band_center = (row_y1 + row_y2) / 2
            
            # Vectorized overlap check for all orphans
            orphan_y1 = orphan_bboxes[:, 1]
            orphan_y2 = orphan_bboxes[:, 3]
            orphan_centers = (orphan_y1 + orphan_y2) / 2
            
            # Check which orphans overlap this row band
            overlap = ((orphan_y1 >= row_y1) & (orphan_y1 <= row_y2)) | \
                     ((orphan_y2 >= row_y1) & (orphan_y2 <= row_y2)) | \
                     ((orphan_y1 <= row_y1) & (orphan_y2 >= row_y2))
            
            if overlap.any():
                # Calculate distances for overlapping orphans
                distances = np.abs(orphan_centers - band_center)
                
                # Store assignments with distances
                for i in np.where(overlap)[0]:
                    oid = orphan_ids[i]
                    if oid not in row_assignments or distances[i] < row_assignments[oid][1]:
                        row_assignments[oid] = (row_idx, distances[i])
        
        # Process columns efficiently using precomputed by_col index
        col_assignments = {}
        for col_idx in range(tab_cols):
            # Get column cells efficiently using index
            col_indices = by_col[col_idx] if col_idx < len(by_col) else []
            col_cells = [table_cells[i] for i in col_indices
                        if "colspan_val" not in table_cells[i] and cell_classes[i] > 1]
            
            if not col_cells:
                continue
                
            # Get column band
            col_x1 = min(c["bbox"][0] for c in col_cells)
            col_x2 = max(c["bbox"][2] for c in col_cells)
            band_center = (col_x1 + col_x2) / 2
            
            # Vectorized overlap check
            orphan_x1 = orphan_bboxes[:, 0]
            orphan_x2 = orphan_bboxes[:, 2]
            orphan_centers = (orphan_x1 + orphan_x2) / 2
            
            overlap = ((orphan_x1 >= col_x1) & (orphan_x1 <= col_x2)) | \
                     ((orphan_x2 >= col_x1) & (orphan_x2 <= col_x2)) | \
                     ((orphan_x1 < col_x1) & (orphan_x2 > col_x2))
            
            if overlap.any():
                distances = np.abs(orphan_centers - band_center)
                
                for i in np.where(overlap)[0]:
                    oid = orphan_ids[i]
                    if oid not in col_assignments or distances[i] < col_assignments[oid][1]:
                        col_assignments[oid] = (col_idx, distances[i])
        
        # Build lookup for existing (row_id, col_id) -> index for O(1) access
        rc_to_idx = {(c["row_id"], c["column_id"]): i for i, c in enumerate(new_table_cells)}
        
        # Create/update table cells for orphans
        for pdf_idx, (oid, bbox) in enumerate(zip(orphan_ids, orphan_bboxes)):
            if oid in row_assignments and oid in col_assignments:
                row_id = row_assignments[oid][0]
                col_id = col_assignments[oid][0]
                
                # O(1) lookup instead of O(n) scan
                key = (row_id, col_id)
                idx = rc_to_idx.get(key)
                
                if idx is not None:
                    # Merge bbox
                    cell = new_table_cells[idx]
                    cell["bbox"] = self._merge_two_bboxes(cell["bbox"], bbox.tolist())
                    table_cell_id = cell["cell_id"]
                else:
                    # Create new cell
                    max_cell_id += 1
                    new_cell = {
                        "bbox": bbox.tolist(),
                        "cell_id": max_cell_id,
                        "column_id": col_id,
                        "label": "body",
                        "row_id": row_id,
                        "cell_class": 2,
                    }
                    rc_to_idx[key] = len(new_table_cells)  # Update lookup
                    new_table_cells.append(new_cell)
                    table_cell_id = max_cell_id
                
                # Add match
                confidence = (row_assignments[oid][1] + col_assignments[oid][1]) / 2
                new_matches[oid] = [{"post": confidence, "table_cell_id": table_cell_id}]
        
        return new_matches, new_table_cells, max_cell_id

    def _clear_pdf_cells(self, pdf_cells):
        r"""
        Clean PDF cells from cells that have an empty string as text

        Parameters
        ----------
        pdf_cells : list of dict
            List of PDF cells as defined by Docling

        Returns
        -------
        new_pdf_cells : list of dict
            updated, cleaned list of pdf_cells
        """
        # Simpler list comprehension with safety for missing/None text
        return [c for c in pdf_cells if c.get("text")]

    def _build_lookup_structures(self, table_cells):
        """Build reusable lookup structures to avoid repeated Python loops."""
        # Index cells by column and row for fast access
        by_col = [[] for _ in range(max((c["column_id"] for c in table_cells), default=-1) + 1)]
        by_row = [[] for _ in range(max((c["row_id"] for c in table_cells), default=-1) + 1)]
        
        for i, cell in enumerate(table_cells):
            by_col[cell["column_id"]].append(i)  # Store indices, not cell copies
            by_row[cell["row_id"]].append(i)
            
        # Precompute cell class array with proper type handling
        cell_classes = np.array([int(cell.get("cell_class", 0)) for cell in table_cells], dtype=np.int16)
        
        return by_col, by_row, cell_classes

    def process(self, matching_details, correct_overlapping_cells=False):
        r"""
        Do post processing, see details in the comments below

        Parameters
        ----------
        matching_details : dictionary
            contains all the necessary information for Docling processing
            already has predictions and initial (IOU) matches

        Returns
        -------
        matching_details : dictionary
            matching_details that contain post-processed matches
        """

        self._log().debug("Start prediction post-processing...")
        table_cells = matching_details["table_cells"]
        pdf_cells = self._clear_pdf_cells(matching_details["pdf_cells"])
        
        # Normalize matches to use int keys (if they come in as strings)
        raw_matches = matching_details["matches"]
        matches = {}
        for k, v in raw_matches.items():
            key = int(k) if isinstance(k, str) else k
            matches[key] = v
            
        # Build lookup structures once
        by_col, by_row, cell_classes = self._build_lookup_structures(table_cells)

        # ------------------------------------------------------------------------------------------
        # -1. If initial (IOU) matches are empty,
        # generate new ones based on intersection over cell

        if not matches:
            self._log().debug(
                "-----------------------------------------------------------------"
            )
            self._log().debug(
                "-   NO INITIAL MATCHES TO POST PROCESS, GENERATING NEW ONES...  -"
            )
            self._log().debug(
                "-----------------------------------------------------------------"
            )
            matches = self._run_intersection_match(
                self._cell_matcher, table_cells, pdf_cells
            )
        
        # ------------------------------------------------------------------------------------------
        # 0. Get minimal grid table dimension (cols/rows) - MOVED UP to get max_cell_id
        tab_columns, tab_rows, max_cell_id = self._get_table_dimension(table_cells)
        if self._log().isEnabledFor(logging.DEBUG):
            self._log().debug(f"COLS {tab_columns}/ ROWS {tab_rows}/ MAX CELL ID {max_cell_id}")
        
        # Build boolean mask for true vectorization AFTER max_cell_id is known
        matched_mask = np.zeros(max_cell_id + 1, dtype=bool)
        for match_list in matches.values():
            for m in match_list:
                cid = m["table_cell_id"]
                if 0 <= cid <= max_cell_id:
                    matched_mask[cid] = True

        good_table_cells = []
        bad_table_cells = []
        new_bad_table_cells = []
        fixed_table_cells = []

        # 1. Get good/bad IOU predicted cells for each structural column (of minimal grid)
        for col in range(tab_columns):
            g1, g2 = self._get_good_bad_cells_in_column(table_cells, col, matched_mask, by_col, cell_classes)
            good_table_cells = g1
            bad_table_cells = g2
            
            if self._log().isEnabledFor(logging.DEBUG):
                self._log().debug(f"COLUMN {col}, Good table cells: {len(good_table_cells)}")
                self._log().debug(f"COLUMN {col}, Bad table cells: {len(bad_table_cells)}")

            # 2. Find alignment of good IOU cells per column
            alignment = self._find_alignment_in_column(good_table_cells)
            if self._log().isEnabledFor(logging.DEBUG):
                self._log().debug(f"COLUMN {col}, Alignment: {alignment}")

            # 3. Get median (according to alignment) "bbox left/middle/right X"
            #    coordinate for good IOU cells, get median* cell size in a column.
            gm1, gm2, gm3, gm4 = self._get_median_pos_size(good_table_cells, alignment)
            median_x = gm1
            # median_y = gm2
            median_width = gm3
            median_height = gm4
            if self._log().isEnabledFor(logging.DEBUG):
                self._log().debug(f"Median good X = {median_x}")

            # 4. Move bad cells to the median* (left/middle/right) good in a column
            # nc = self._move_cells_to_left_pos(bad_table_cells, median_x, True,
            # TODO:
            nc = self._move_cells_to_left_pos(
                bad_table_cells, median_x, False, median_width, median_height, alignment
            )
            new_bad_table_cells = nc
            fixed_table_cells.extend(good_table_cells)
            fixed_table_cells.extend(new_bad_table_cells)

        # ====================================================================================
        # Sort table_cells by cell_id before running IOU, to have correct indexes on the output
        fixed_table_cells_sorted = sorted(fixed_table_cells, key=lambda k: k["cell_id"])

        # 5. Generate new matches, run Intersection over cell(pdf) on a table cells
        ip = self._run_intersection_match(
            self._cell_matcher, fixed_table_cells_sorted, pdf_cells
        )
        intersection_pdf_matches = ip

        # 6. NOT USED

        # 7. De-duplicate columns in aligned_table_cells
        # according to highest column score in: matches + intersection_pdf_matches
        # (this is easier now, because duplicated cells will have same bboxes)
        dd1, dd2, dd3 = self._deduplicate_cells(
            tab_columns, fixed_table_cells_sorted, matches, intersection_pdf_matches
        )
        dedupl_table_cells = dd1
        dedupl_matches = dd2
        new_tab_columns = dd3  # CRITICAL: Use the updated column count

        if self._log().isEnabledFor(logging.DEBUG):
            self._log().debug("Deduplication complete...")

        # 8. Do final assignment of table bbox to pdf cell based on saved scores,
        # preferring IOU over PDF Intersection, and higher Intersection over lower
        # ! IOU matches currently disabled,
        # and final assigment is done only on IOC matches
        final_matches = self._do_final_asignment(
            dedupl_table_cells, matches, dedupl_matches
        )

        # 8.a. Re-align bboxes / re-run matching
        dedupl_table_cells_sorted = sorted(
            dedupl_table_cells, key=lambda k: k["cell_id"]
        )

        # Make align cutoff configurable
        ALIGN_CAP = int(self._config.get("align_pdf_cost_cap", 300))
        if len(pdf_cells) > ALIGN_CAP:
            # For performance, skip this step if there are too many pdf_cells
            aligned_table_cells2 = dedupl_table_cells_sorted
        else:
            aligned_table_cells2 = self._align_table_cells_to_pdf(
                dedupl_table_cells_sorted, pdf_cells, final_matches
            )

        # CRITICAL: Rebuild lookups for the FINAL table_cells (indices changed after dedup/align)
        by_col2, by_row2, cell_classes2 = self._build_lookup_structures(aligned_table_cells2)

        # 9. Distance-match orphans - USE OPTIMIZED VERSION with correct column count and structures
        po1, po2, po3 = self._pick_orphan_cells_vectorized(
            tab_rows,
            new_tab_columns,  # Use deduplicated column count
            max_cell_id,
            aligned_table_cells2,
            pdf_cells,
            final_matches,
            by_col2, by_row2, cell_classes2  # Use correct structures for aligned cells
        )
        final_matches_wo = po1
        table_cells_wo = po2
        max_cell_id = po3

        if correct_overlapping_cells:
            # As the last step - correct cell bboxes in a way that they don't overlap:
            if len(table_cells_wo) <= 300:  # For performance reasons
                table_cells_wo = self._find_overlapping(table_cells_wo)

        # Guard large debug dumps
        if self._log().isEnabledFor(logging.DEBUG):
            self._log().debug("*** final_matches_wo")
            self._log().debug(final_matches_wo)
            self._log().debug("*** table_cells_wo")
            self._log().debug(table_cells_wo)

        # Fixed: iterate actual items, not range
        if self._log().isEnabledFor(logging.DEBUG):
            for pdf_cell_id, pdf_cell_match in final_matches_wo.items():
                if len(pdf_cell_match) > 1:
                    self._log().info(f"!!! Multiple - {len(pdf_cell_match)}x pdf cell match with id: {pdf_cell_id}")
                if pdf_cell_match:
                    tcellid = pdf_cell_match[0]["table_cell_id"]
                    for tcell in table_cells_wo:
                        if tcell["cell_id"] == tcellid:
                            mrow = tcell["row_id"]
                            mcol = tcell["column_id"]
                            self._log().debug(f"pdf cell: {pdf_cell_id} -> row: {mrow} | col: {mcol}")
                            break

        matching_details["table_cells"] = table_cells_wo
        matching_details["matches"] = final_matches_wo
        matching_details["pdf_cells"] = pdf_cells

        self._log().debug("Done prediction matching and post-processing!")
        return matching_details