import sys
from typing import List
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.page import TextCell
from pydantic import BaseModel, field_serializer, FieldSerializationInfo, PrivateAttr

from docling.datamodel.base_models import BoundingBox,  PydanticSerCtxKey, round_pydantic_float


class Cluster(BaseModel):
    id: int
    label: DocItemLabel
    bbox: BoundingBox
    confidence: float = 1.0
    cells: List[TextCell] = []
    children: List["Cluster"] = []  # Add child cluster support

    _first_cell_index: int = PrivateAttr(default=sys.maxsize)

    @field_serializer("confidence")
    def _serialize(self, value: float, info: FieldSerializationInfo) -> float:
        return round_pydantic_float(value, info.context, PydanticSerCtxKey.CONFID_PREC)
