import numpy as np
from PIL import Image
from pydantic import BaseModel
from typing import Optional, Dict, List

from docling_core.types.doc import CoordOrigin
from docling_core.types.doc import TextCell, BoundingBox
from docling_core.types.doc.page import TextCellUnit
from docling.datamodel.base_models import Size, ConfigDict, SegmentedPdfPage, PagePredictions, AssembledUnit

TOK_DTYPE = np.dtype([('id','i4'),('l','f4'),('t','f4'),('r','f4'),('b','f4')])


class Page(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    page_no: int
    # page_hash: Optional[str] = None
    size: Optional[Size] = None
    parsed_page: Optional[SegmentedPdfPage] = None
    predictions: PagePredictions = PagePredictions()
    assembled: Optional[AssembledUnit] = None

    _backend: Optional["PdfPageBackend"] = (
        None  # Internal PDF backend. By default it is cleared during assembling.
    )
    _default_image_scale: float = 1.0  # Default image scale for external usage.
    _image_cache: Dict[
        float, Image
    ] = {}  # Cache of images in different scales. By default it is cleared during assembling.
    _np_image_cache: dict[float, np.ndarray] = {}

    tokens_np: np.array = None

    # token_index: Optional[PageTokenIndex] = None
    #
    # def build_token_index(self):
    #     self.token_index = PageTokenIndex(scale=2.0, page_height=self.size.height, page_width=self.size.width)
    #     self.token_index.build(self.parsed_page, grid_cell=256)

    @property
    def cells(self) -> List[TextCell]:
        """Return text cells as a read-only view of parsed_page.textline_cells."""
        if self.parsed_page is not None:
            return self.parsed_page.textline_cells
        else:
            return []

    def get_image(
            self,
            scale: float = 1.0,
            max_size: Optional[int] = None,
            cropbox: Optional[BoundingBox] = None,
    ) -> Optional[Image]:
        if self._backend is None:
            return self._image_cache.get(scale, None)

        if max_size:
            assert self.size is not None
            scale = min(scale, max_size / max(self.size.as_tuple()))

        if scale not in self._image_cache:
            if cropbox is None:
                self._image_cache[scale] = self._backend.get_page_image(scale=scale)
            else:
                return self._backend.get_page_image(scale=scale, cropbox=cropbox)

        if cropbox is None:
            return self._image_cache[scale]
        else:
            page_im = self._image_cache[scale]
            assert self.size is not None
            return page_im.crop(
                cropbox.to_top_left_origin(page_height=self.size.height)
                .scaled(scale=scale)
                .as_tuple()
            )

    def get_image_np(self, scale: float = 1.0) -> np.ndarray:
        arr = self._np_image_cache.get(scale)
        if arr is not None:
            return arr
        pil = self.get_image(scale=scale)  # hits _image_cache if pre-rendered
        arr = np.asarray(pil, dtype=np.uint8)  # HWC, uint8
        if not arr.flags['C_CONTIGUOUS']:
            arr = np.ascontiguousarray(arr)  # make it once, reuse forever
        self._np_image_cache[scale] = arr
        return arr

    @property
    def image(self) -> Optional[Image]:
        return self.get_image(scale=self._default_image_scale)


def build_tokens_np(parsed_page, page_size) -> np.ndarray:
    """
    Build a page-wide tokens array in TOP-LEFT origin, UNscaled.
    Works whether SegmentedPdfPage exposes get_cells() or only get_cells_in_bbox().
    """
    # 1) Fetch WORD cells page-wide
    try:
        # Newer API
        cells = parsed_page.get_cells(cell_unit=TextCellUnit.WORD)
    except TypeError:
        # Older API expects a bbox
        full = BoundingBox(
            l=0.0, b=0.0, r=float(page_size.width), t=float(page_size.height),
            coord_origin=CoordOrigin.BOTTOMLEFT
        )
        cells = parsed_page.get_cells_in_bbox(
            cell_unit=TextCellUnit.WORD,
            bbox=full,
            ios=0.0
        )

    if not cells:
        # Fallback to textlines if WORDS are empty
        cells = getattr(parsed_page, "textline_cells", [])

    # 2) Pack into a compact NumPy struct
    out = np.empty(max(1, len(cells)), dtype=TOK_DTYPE)
    k = 0
    H = float(page_size.height)

    for c in cells:
        text = (c.text or "").strip()
        if not text:
            continue
        bb_tl = c.rect.to_top_left_origin(page_height=H).to_bounding_box()
        out[k]['id'] = int(getattr(c, 'index', k))
        out[k]['l']  = float(bb_tl.l)
        out[k]['t']  = float(bb_tl.t)
        out[k]['r']  = float(bb_tl.r)
        out[k]['b']  = float(bb_tl.b)
        k += 1

    return np.ascontiguousarray(out[:k])
