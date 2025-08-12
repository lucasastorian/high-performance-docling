from typing import List

from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.models.readingorder_model import ReadingOrderModel, ReadingOrderOptions
from docling.datamodel.base_models import AssembledUnit
from docling_core.types.doc import ImageRef, PictureItem

from page_assemble_model import PageAssembleModel, PageAssembleOptions


class DocumentAssembler:
    """Final assembly - can run on Lambda or GPU."""

    def __init__(self, pipeline_options=None):
        self.pipeline_options = pipeline_options or PdfPipelineOptions()
        self.page_assembler = PageAssembleModel(options=PageAssembleOptions())
        self.reading_order_model = ReadingOrderModel(options=ReadingOrderOptions())

    def assemble_document(self, input_doc, processed_pages: List[Page]) -> ConversionResult:
        """Assemble final document from processed pages."""

        print(f"\n📚 Assembling document from {len(processed_pages)} pages...")

        # Create ConversionResult
        conv_res = ConversionResult(input=input_doc)
        conv_res.pages = processed_pages

        # Run page assembly
        print("  1️⃣ Assembling page elements...")
        assembled_pages = list(self.page_assembler(conv_res, processed_pages))
        conv_res.pages = assembled_pages

        # Collect all elements
        all_elements = []
        all_headers = []
        all_body = []

        for page in conv_res.pages:
            if page.assembled:
                all_elements.extend(page.assembled.elements)
                all_headers.extend(page.assembled.headers)
                all_body.extend(page.assembled.body)

        conv_res.assembled = AssembledUnit(
            elements=all_elements,
            headers=all_headers,
            body=all_body
        )

        # Build final document with reading order
        print("  2️⃣ Determining reading order...")
        conv_res.document = self.reading_order_model(conv_res)

        if self.pipeline_options.generate_picture_images:
            print("  3️⃣ Extracting figure images...")
            self._extract_figure_images(conv_res)

        print("✅ Document assembly complete!")
        return conv_res

    def _extract_figure_images(self, conv_res: ConversionResult):
        """Extract actual images for figure elements."""
        scale = self.pipeline_options.images_scale

        for element, _level in conv_res.document.iterate_items():
            if isinstance(element, PictureItem) and len(element.prov) > 0:
                # Get the page containing this figure
                page_ix = element.prov[0].page_no - 1
                page = next(
                    (p for p in conv_res.pages if p.page_no == page_ix),
                    None
                )

                if page and page.image:
                    # Crop the figure from the page image
                    crop_bbox = (
                        element.prov[0]
                        .bbox.scaled(scale=scale)
                        .to_top_left_origin(
                            page_height=page.size.height * scale
                        )
                    )

                    cropped_im = page.image.crop(crop_bbox.as_tuple())
                    element.image = ImageRef.from_pil(
                        cropped_im,
                        dpi=int(72 * scale)
                    )
                    print(f"     Extracted image for figure on page {page.page_no}")
