from typing import List
from docling.datamodel.base_models import Page
from docling.datamodel.document import InputDocument
from docling.datamodel.pipeline_options import PdfPipelineOptions


class LambdaPreprocessor:
    """Runs on Lambda - just basic preprocessing, no OCR."""

    def __init__(self, pipeline_options=None):
        self.pipeline_options = pipeline_options or PdfPipelineOptions()
        self.images_scale = self.pipeline_options.images_scale

    def process_batch(self, input_doc: InputDocument, start: int, end: int) -> List[Page]:
        """Process a batch of pages."""
        pages = []

        for page_no in range(start, min(end + 1, input_doc.page_count)):
            page = Page(page_no=page_no)

            # Initialize backend
            page._backend = input_doc._backend.load_page(page.page_no)
            if not page._backend or not page._backend.is_valid():
                continue

            # Basic setup
            page.size = page._backend.get_size()
            page.parsed_page = page._backend.get_segmented_page()

            # Cache images
            page.get_image(scale=1.0)
            if self.images_scale != 1.0:
                page._default_image_scale = self.images_scale
                page.get_image(scale=self.images_scale)

            pages.append(page)

        print(f"Lambda preprocessed pages {start}-{end}")
        return pages


def distribute_preprocessing(input_doc, batch_size=10):
    """Simulate distributed Lambda preprocessing."""
    all_pages = []
    preprocessor = LambdaPreprocessor()

    for start in range(0, input_doc.page_count, batch_size):
        end = min(start + batch_size - 1, input_doc.page_count - 1)

        batch_pages = preprocessor.process_batch(input_doc, start, end)
        all_pages.extend(batch_pages)

    return all_pages
