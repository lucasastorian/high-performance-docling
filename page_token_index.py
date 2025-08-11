import numpy as np
from typing import Optional

from docling_core.types.doc import BoundingBox
from docling.datamodel.base_models import Size, SegmentedPdfPage

TOK_DTYPE = np.dtype([('id', 'i4'), ('l', 'f4'), ('t', 'f4'), ('r', 'f4'), ('b', 'f4')])


class PageTokenIndex:

    def __init__(self, parsed_page: SegmentedPdfPage, size: Size):
        self.parsed_page = parsed_page
        self.size = size

        self._tokens_np: Optional[np.ndarray] = None  # (N,), TOK_DTYPE
        self._grid_keys: Optional[np.ndarray] = None  # (M,2) int32, unique (gx,gy) per cell in postings table
        self._grid_indptr: Optional[np.ndarray] = None  # (M+1,) int32, CSR-style offsets
        self._grid_postings: Optional[np.ndarray] = None  # (P,) int32, concatenated token indices
        self._grid_cell: int = 256

    def build(self, scale: float = 2.0, grid_cell: int = 256) -> None:
        """
        Build a page-level token array (top-left origin, scaled) and a uniform grid index.
        Safe to call multiple times; replaces prior index.
        """
        assert self.parsed_page is not None and self.size is not None
        self._grid_cell = int(grid_cell)

        # 1) Extract tokens once from parsed_page (WORD level preferred)
        #    Convert to top-left origin and scale to TF coords up-front.
        toks = []
        sx = sy = float(scale)
        H = float(self.size.height)

        # WORD-level is best; fallback to textline if empty
        from docling_core.types.doc.page import TextCellUnit
        cells = self.parsed_page.get_cells_in_bbox(
            cell_unit=TextCellUnit.WORD,
            bbox=BoundingBox(l=0, t=0, r=self.size.width, b=self.size.height),
            ios=0.0
        )
        if not cells:
            cells = self.parsed_page.textline_cells

        toks_reserve = max(1024, len(cells))
        toks = np.empty(toks_reserve, dtype=TOK_DTYPE)
        k = 0

        for c in cells:
            text = c.text.strip()
            if not text:
                continue
            bb = c.rect.to_top_left_origin(page_height=self.size.height).to_bounding_box()
            # scale to TF space now (avoid multiplies during queries)
            toks[k]['id'] = int(getattr(c, 'index', k))
            toks[k]['l'] = bb.l * sx
            toks[k]['t'] = bb.t * sy
            toks[k]['r'] = bb.r * sx
            toks[k]['b'] = bb.b * sy
            k += 1

        self._tokens_np = np.ascontiguousarray(toks[:k])  # trim & ensure C-contig

        # 2) Build a compact uniform grid (CSR-like)
        if k == 0:
            self._grid_keys = np.zeros((0, 2), dtype=np.int32)
            self._grid_indptr = np.zeros((1,), dtype=np.int32)
            self._grid_postings = np.zeros((0,), dtype=np.int32)
            return

        # grid keys per token (gx, gy) from token center
        cx = 0.5 * (self._tokens_np['l'] + self._tokens_np['r'])
        cy = 0.5 * (self._tokens_np['t'] + self._tokens_np['b'])
        gx = np.floor_divide(cx.astype(np.int32), self._grid_cell)
        gy = np.floor_divide(cy.astype(np.int32), self._grid_cell)
        keys = np.stack([gx, gy], axis=1)

        # sort by (gx,gy) to build postings
        order = np.lexsort((keys[:, 1], keys[:, 0]))
        keys_sorted = keys[order]
        # unique keys and segment sizes
        uniq, idx_start = np.unique(keys_sorted, axis=0, return_index=True)
        sizes = np.diff(np.append(idx_start, keys_sorted.shape[0]))
        indptr = np.empty(uniq.shape[0] + 1, dtype=np.int32)
        indptr[0] = 0
        np.cumsum(sizes, out=indptr[1:])

        postings = order.astype(np.int32)  # token indices per cell, concatenated

        self._grid_keys = np.ascontiguousarray(uniq.astype(np.int32))
        self._grid_indptr = indptr
        self._grid_postings = postings

    def _grid_cell_hits(self, lb: float, tb: float, rb: float, bb: float) -> np.ndarray:
        """
        Return candidate token indices by union of grid cells intersecting bbox.
        """
        if (self._grid_keys is None or self._grid_indptr is None or
                self._grid_postings is None or self._tokens_np is None):
            # No grid -> all tokens
            return np.arange(self._tokens_np.shape[0], dtype=np.int32)

        G = self._grid_cell
        gx0 = int(lb // G)
        gy0 = int(tb // G)
        gx1 = int(rb // G)
        gy1 = int(bb // G)

        if gx1 < gx0 or gy1 < gy0:
            return np.empty((0,), dtype=np.int32)

        # Build the set of (gx,gy) covered
        wanted = np.array([(gx, gy)
                           for gx in range(gx0, gx1 + 1)
                           for gy in range(gy0, gy1 + 1)],
                          dtype=np.int32)
        if wanted.size == 0:
            return np.empty((0,), dtype=np.int32)

        # Pack keys to 64-bit for a single searchsorted domain
        def pack(a: np.ndarray) -> np.ndarray:
            # a: (..., 2) int32, assumed non-negative
            return (a[:, 0].astype(np.int64) << 32) | (a[:, 1].astype(np.int64) & 0xffffffff)

        K = pack(self._grid_keys)  # sorted by construction
        W = pack(wanted)  # not necessarily sorted, but we don’t need it sorted

        # For each wanted key, find insertion position in K
        pos = np.searchsorted(K, W)

        # IMPORTANT: mask out out-of-range BEFORE indexing K[pos]
        in_bounds = (pos < K.size)
        if not in_bounds.any():
            return np.empty((0,), dtype=np.int32)

        pos_ib = pos[in_bounds]
        W_ib = W[in_bounds]

        # Now safe to compare
        is_hit = (K[pos_ib] == W_ib)
        if not is_hit.any():
            return np.empty((0,), dtype=np.int32)

        hit_pos = pos_ib[is_hit]

        starts = self._grid_indptr[hit_pos]
        ends = self._grid_indptr[hit_pos + 1]

        total = int(np.sum(ends - starts))
        if total == 0:
            return np.empty((0,), dtype=np.int32)

        out = np.empty((total,), dtype=np.int32)
        off = 0
        for s, e in zip(starts, ends):
            n = e - s
            out[off:off + n] = self._grid_postings[s:e]
            off += n

        # Deduplicate in case bbox covers multiple grid cells that share tokens
        if out.size <= 1:
            return out
        return np.unique(out)

    def query_tokens_in_bbox(self, bbox_scaled_tl: BoundingBox, ios: float = 0.8) -> np.ndarray:
        """
        Vectorized IOS filter in TF scale + TL origin. Returns token indices.
        """
        assert self._tokens_np is not None
        lb, tb, rb, bb = bbox_scaled_tl.as_tuple()  # already scaled & top-left

        cand = self._grid_cell_hits(lb, tb, rb, bb)

        if cand.size == 0:
            return cand

        toks = self._tokens_np[cand]
        L, T, R, B = toks['l'], toks['t'], toks['r'], toks['b']

        inter_w = np.maximum(0.0, np.minimum(R, rb) - np.maximum(L, lb))
        inter_h = np.maximum(0.0, np.minimum(B, bb) - np.maximum(T, tb))
        inter = inter_w * inter_h
        area = (R - L) * (B - T)
        keep = inter >= (ios * area)

        return cand[keep]
