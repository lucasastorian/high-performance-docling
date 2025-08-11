# token_index_tf.py
from dataclasses import dataclass
import numpy as np
from docling_core.types.doc import BoundingBox

# (id, l,t,r,b) in **TF space** (scaled, top-left)
TOK_DTYPE = np.dtype([
    ('id', 'i4'),
    ('l',  'f4'),
    ('t',  'f4'),
    ('r',  'f4'),
    ('b',  'f4'),
])


@dataclass
class PageTokenIndex:

    scale: float
    page_height: float
    page_width: float
    _tokens: np.ndarray | None = None         # (N,), TOK_DTYPE  in TL+scaled
    _texts: list[str] | None = None
    _grid_keys: np.ndarray | None = None      # (M,2) int32 (gx,gy) present
    _grid_indptr: np.ndarray | None = None    # (M+1,)
    _grid_postings: np.ndarray | None = None  # (P,)
    _cell: int = 256                          # grid cell in TF px

    def build(self, parsed_page, grid_cell: int = 256):
        """
        Extract WORD tokens, convert to **top-left origin**, scale by self.scale, build uniform grid.
        """
        from docling_core.types.doc.page import TextCellUnit

        self._cell = int(grid_cell)
        sx = sy = float(self.scale)
        H = float(self.page_height)

        full_page_bbox = BoundingBox(
            l=0.0, t=0.0, r=self.page_width, b=self.page_height  # TOP-LEFT origin
        )

        cells = parsed_page.get_cells_in_bbox(
            cell_unit=TextCellUnit.WORD,
            bbox=full_page_bbox,
            ios=0.0,
        )

        if not cells:
            cells = parsed_page.textline_cells

        N = max(1024, len(cells))
        toks = np.empty(N, dtype=TOK_DTYPE)
        texts: list[str] = []
        k = 0

        for c in cells:
            txt = (c.text or "").strip()
            if not txt:
                continue
            # -> TL origin, then scale (matches your old code semantics)
            bb = c.rect.to_top_left_origin(page_height=H).to_bounding_box()
            l, t, r, b = bb.l * sx, bb.t * sy, bb.r * sx, bb.b * sy
            if r <= l or b <= t:
                continue
            if k == toks.shape[0]:
                toks = np.resize(toks, int(toks.shape[0] * 1.5))
            toks[k]['id'] = int(getattr(c, 'index', k))
            toks[k]['l']  = l; toks[k]['t'] = t; toks[k]['r'] = r; toks[k]['b'] = b
            texts.append(txt)
            k += 1

        self._tokens = np.ascontiguousarray(toks[:k])
        self._texts = texts

        # build grid (on token centers in TF space)
        if k == 0:
            self._grid_keys = np.zeros((0,2), np.int32)
            self._grid_indptr = np.zeros((1,), np.int32)
            self._grid_postings = np.zeros((0,), np.int32)
            return

        cx = 0.5 * (self._tokens['l'] + self._tokens['r'])
        cy = 0.5 * (self._tokens['t'] + self._tokens['b'])
        gx = np.floor_divide(cx.astype(np.int32), self._cell)
        gy = np.floor_divide(cy.astype(np.int32), self._cell)
        keys = np.stack([gx, gy], axis=1)

        order = np.lexsort((keys[:,1], keys[:,0]))
        ks = keys[order]
        uniq, idx_start = np.unique(ks, axis=0, return_index=True)
        sizes = np.diff(np.append(idx_start, ks.shape[0]))

        indptr = np.empty(uniq.shape[0] + 1, dtype=np.int32)
        indptr[0] = 0
        np.cumsum(sizes, out=indptr[1:])

        self._grid_keys = uniq.astype(np.int32)
        self._grid_indptr = indptr
        self._grid_postings = order.astype(np.int32)

    # ---- query in TF (TL origin) ----

    def _candidates(self, l, t, r, b):
        if self._grid_keys is None:
            return np.arange(self._tokens.shape[0], dtype=np.int32)
        G = self._cell
        gx0 = int(l // G); gx1 = int(r // G)
        gy0 = int(t // G); gy1 = int(b // G)
        if gx1 < gx0 or gy1 < gy0:
            return np.empty((0,), dtype=np.int32)

        wanted = np.array([(gx, gy)
                           for gx in range(gx0, gx1+1)
                           for gy in range(gy0, gy1+1)], dtype=np.int32)
        if wanted.size == 0:
            return np.empty((0,), dtype=np.int32)

        def pack(a):  # pack (gx,gy) → sortable 64-bit
            return (a[:,0].astype(np.int64) << 32) ^ (a[:,1].astype(np.int64) & 0xffffffff)

        K = pack(self._grid_keys)        # sorted by construction
        W = pack(wanted)
        pos = np.searchsorted(K, W)
        mask = (pos < K.size) & (K[pos] == W)
        pos = pos[mask]
        if pos.size == 0:
            return np.empty((0,), dtype=np.int32)

        starts = self._grid_indptr[pos]
        ends   = self._grid_indptr[pos+1]
        total  = int(np.sum(ends - starts))
        out = np.empty((total,), dtype=np.int32)
        off = 0
        for s,e in zip(starts, ends):
            n = e - s
            out[off:off+n] = self._grid_postings[s:e]
            off += n
        return np.unique(out)

    def query_tokens_ios(self, bbox_tl_scaled, ios: float = 0.0) -> np.ndarray:
        """
        bbox_tl_scaled: (l,t,r,b) in TF space (TL origin), same as your current `_get_table_tokens`.
        Returns token indices with Intersection-Over-Self >= ios (0.0 → any overlap).
        """
        l,t,r,b = map(float, bbox_tl_scaled)
        if r <= l or b <= t or self._tokens is None:
            return np.empty((0,), dtype=np.int32)

        cand = self._candidates(l,t,r,b)
        if cand.size == 0:
            return cand

        toks = self._tokens[cand]
        L,T,R,B = toks['l'], toks['t'], toks['r'], toks['b']
        inter_w = np.maximum(0.0, np.minimum(R, r) - np.maximum(L, l))
        inter_h = np.maximum(0.0, np.minimum(B, b) - np.maximum(T, t))
        inter = inter_w * inter_h
        area  = (R - L) * (B - T)

        keep = inter > 0.0 if ios <= 0.0 else inter >= (ios * area)
        return cand[keep]
