import numpy as np
from docling_core.types.doc.base import CoordOrigin


class PageWordIndex:
    __slots__ = ("l","t","r","b","ids","texts","H","W","gx","gy","grid",
                 "area","id_arr","id_to_idx")

    def __init__(self, sp, unit, page_height, grid_nx=None, grid_ny=None):
        # Collect cells
        cells = list(sp.iterate_cells(unit))
        N = len(cells)
        if N == 0:
            self.l = self.t = self.r = self.b = np.empty((0,), np.float32)
            self.ids = []; self.texts = []; self.H = self.W = 0
            self.gx = self.gy = 0; self.grid = []
            # new fields
            self.area = np.empty((0,), np.float32)
            self.id_arr = np.empty((0,), np.int64)
            self.id_to_idx = {}
            return

        # Normalize to TOPLEFT once; compute AABBs without Pydantic objects
        L = np.empty(N, np.float32); T = np.empty(N, np.float32)
        R = np.empty(N, np.float32); B = np.empty(N, np.float32)
        ids = np.empty(N, np.int64); texts = [None]*N

        for i, c in enumerate(cells):
            r = c.rect
            if r.coord_origin == CoordOrigin.TOPLEFT:
                ty = min(r.r_y0, r.r_y1, r.r_y2, r.r_y3)
                by = max(r.r_y0, r.r_y1, r.r_y2, r.r_y3)
            else:  # BOTTOMLEFT
                ty = page_height - max(r.r_y0, r.r_y1, r.r_y2, r.r_y3)
                by = page_height - min(r.r_y0, r.r_y1, r.r_y2, r.r_y3)

            lx = min(r.r_x0, r.r_x1, r.r_x2, r.r_x3)
            rx = max(r.r_x0, r.r_x1, r.r_x2, r.r_x3)

            L[i], T[i], R[i], B[i] = lx, ty, rx, by
            ids[i] = int(c.index)
            texts[i] = c.text

        self.l, self.t, self.r, self.b = L, T, R, B
        self.ids, self.texts = list(ids.tolist()), texts  # keep legacy types

        # NEW: vector fields and map
        self.area = (R - L) * (B - T)
        self.id_arr = ids
        # dict comprehension is fine at this scale
        self.id_to_idx = {int(pid): int(i) for i, pid in enumerate(ids)}

        # Page size (crop box)
        self.W = sp.dimension.crop_bbox.width
        self.H = sp.dimension.crop_bbox.height

        # Grid sizing heuristic: ≈ sqrt(N)/2 per axis, clamp to [8, 64]
        if grid_nx is None or grid_ny is None:
            g = max(8, min(64, int(np.sqrt(max(N,1)) // 2 or 8)))
            grid_nx = grid_nx or g
            grid_ny = grid_ny or g
        self.gx, self.gy = grid_nx, grid_ny
        self.grid = [[] for _ in range(self.gx * self.gy)]

        # Assign each word to all bins it overlaps (prevents edge misses)
        cell_w = self.W / self.gx
        cell_h = self.H / self.gy
        ix0 = np.clip((L / cell_w).astype(int), 0, self.gx-1)
        ix1 = np.clip(((R - 1e-6) / cell_w).astype(int), 0, self.gx-1)
        iy0 = np.clip((T / cell_h).astype(int), 0, self.gy-1)
        iy1 = np.clip(((B - 1e-6) / cell_h).astype(int), 0, self.gy-1)

        for i in range(N):
            for gx in range(ix0[i], ix1[i] + 1):
                row = gx * self.gy
                for gy in range(iy0[i], iy1[i] + 1):
                    self.grid[row + gy].append(i)

    # --- Legacy API (unchanged) ---
    def query_bbox(self, ql, qt, qr, qb, ios=0.8, scale=1.0):
        if not self.ids:
            return []

        cell_w = self.W / self.gx
        cell_h = self.H / self.gy
        gx0 = int(np.clip(ql / cell_w, 0, self.gx-1))
        gx1 = int(np.clip((qr - 1e-6) / cell_w, 0, self.gx-1))
        gy0 = int(np.clip(qt / cell_h, 0, self.gy-1))
        gy1 = int(np.clip((qb - 1e-6) / cell_h, 0, self.gy-1))

        cand = []
        for gx in range(gx0, gx1 + 1):
            row = gx * self.gy
            for gy in range(gy0, gy1 + 1):
                cand.extend(self.grid[row + gy])
        if not cand:
            return []

        cand = np.unique(np.fromiter(cand, dtype=np.int32))
        L, T, R, B = self.l[cand], self.t[cand], self.r[cand], self.b[cand]

        iw = np.maximum(0.0, np.minimum(R, qr) - np.maximum(L, ql))
        ih = np.maximum(0.0, np.minimum(B, qb) - np.maximum(T, qt))
        inter = iw * ih
        keep = inter / ( (R-L)*(B-T) + 1e-6 ) >= ios
        if not np.any(keep):
            return []

        cand = cand[keep]
        L, T, R, B = L[keep], T[keep], R[keep], B[keep]

        out = []
        s = float(scale)
        for i, l, t, r, b in zip(cand, L, T, R, B):
            text = self.texts[i]
            if not text or text.isspace():
                continue
            if text[0].isspace() or text[-1].isspace():
                text = text.strip()
            out.append({
                "id": self.ids[i],
                "text": text,
                "bbox": {"l": float(l*s), "t": float(t*s),
                         "r": float(r*s), "b": float(b*s)}
            })
        return out

    # --- New fast paths (non-breaking) ---

    def query_bbox_idx(self, ql, qt, qr, qb, ios=0.8):
        """Return indices of words overlapping the bbox by IoS>=ios (TOP-LEFT)."""
        if self.gx == 0 or self.gy == 0:
            return np.empty((0,), dtype=np.int32)

        cell_w = self.W / self.gx
        cell_h = self.H / self.gy
        gx0 = int(np.clip(ql / cell_w, 0, self.gx-1))
        gx1 = int(np.clip((qr - 1e-6) / cell_w, 0, self.gx-1))
        gy0 = int(np.clip(qt / cell_h, 0, self.gy-1))
        gy1 = int(np.clip((qb - 1e-6) / cell_h, 0, self.gy-1))

        cand = []
        for gx in range(gx0, gx1 + 1):
            row = gx * self.gy
            for gy in range(gy0, gy1 + 1):
                cand.extend(self.grid[row + gy])
        if not cand:
            return np.empty((0,), dtype=np.int32)

        cand = np.unique(np.fromiter(cand, dtype=np.int32))
        L, T, R, B = self.l[cand], self.t[cand], self.r[cand], self.b[cand]

        iw = np.maximum(0.0, np.minimum(R, qr) - np.maximum(L, ql))
        ih = np.maximum(0.0, np.minimum(B, qb) - np.maximum(T, qt))
        inter = iw * ih
        denom = self.area[cand] + 1e-6
        keep = inter / denom >= ios
        return cand[keep]

    def intersect_iopdf_idx(self, indices, x1, y1, x2, y2):
        """Compute intersection-over-PDFcell for a set of word indices."""
        if indices.size == 0:
            return np.empty((0,), dtype=np.float32)
        i = indices
        iw = np.maximum(0.0, np.minimum(self.r[i], x2) - np.maximum(self.l[i], x1))
        ih = np.maximum(0.0, np.minimum(self.b[i], y2) - np.maximum(self.t[i], y1))
        inter = iw * ih
        return inter / np.maximum(self.area[i], 1e-6)

    def pack_tokens(self, indices, scale=1.0, trim=True):
        """Build legacy token dicts for given indices (lazy packing)."""
        if indices.size == 0:
            return []
        s = float(scale)
        out = []
        for i in indices.tolist():
            txt = self.texts[i] or ""
            if trim:
                txt = txt.strip()
                if not txt:
                    continue
            out.append({
                "id": int(self.id_arr[i]),
                "text": txt,
                "bbox": {"l": float(self.l[i]*s), "t": float(self.t[i]*s),
                         "r": float(self.r[i]*s), "b": float(self.b[i]*s)}
            })
        return out

    def query_bbox_bulk(self, bboxes, ios=0.8):
        """
        Vectorized multi-bbox query.
        bboxes: array-like shape (M,4) of [l,t,r,b] in TOP-LEFT.
        Returns: list of np.ndarray indices, length M.
        """
        out = []
        for (ql,qt,qr,qb) in bboxes:
            out.append(self.query_bbox_idx(float(ql), float(qt), float(qr), float(qb), ios=ios))
        return out
