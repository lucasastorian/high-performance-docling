import numpy as np
from PIL import Image
from pydantic import BaseModel
from typing import Optional, Dict, List

from docling_core.types.doc import TextCell, BoundingBox
from docling.datamodel.base_models import Size, ConfigDict, SegmentedPdfPage, PagePredictions, AssembledUnit
from page_token_index import PageTokenIndex


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

    token_index: Optional[PageTokenIndex] = None

    def build_token_index(self):
        self.token_index = PageTokenIndex(page_height=self.size.height, page_width=self.size.width)
        self.token_index.build(self.parsed_page, grid_cell=256)

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
