import os
import re
import torch
import time
from typing import List
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, LayoutOptions, TableStructureOptions
)
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.page import TextCell, BoundingRectangle
import numpy as np

from optimized.layout.layout_model import LayoutModel
from standard.table_structure_model import TableStructureModel
from optimized.table.table_timing_debug import print_timing_summary
from table_regression_runner import TableRegressionRunner, Tolerances


def fmt_secs(s: float) -> str:
    """Pretty print durations: e.g. 85.2 ms, 2.31 s."""
    return f"{s * 1000:.1f} ms" if s < 1 else f"{s:.2f} s"


class GPUProcessor:
    """Runs on GPU - Layout + Smart OCR + Table Analysis."""

    def __init__(self, pipeline_options=None):
        self.pipeline_options = pipeline_options or PdfPipelineOptions()
        self.pipeline_options.generate_picture_images = True
        self.pipeline_options.generate_page_images = True

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize Layout Model
        self.layout_model = LayoutModel(
            artifacts_path=None,
            accelerator_options=pipeline_options.accelerator_options,
            options=pipeline_options.layout_options or LayoutOptions()
        )

        # Initialize TrOCR for batched OCR
        self.ocr_processor = TrOCRProcessor.from_pretrained(
            'microsoft/trocr-base-printed',
            use_fast=True
        )
        self.ocr_model = VisionEncoderDecoderModel.from_pretrained(
            'microsoft/trocr-base-printed'
        ).to(device)

        # Initialize Table Model
        self.table_model = TableStructureModel(
            enabled=pipeline_options.do_table_structure,
            artifacts_path=None,
            options=pipeline_options.table_structure_options or TableStructureOptions(),
            accelerator_options=pipeline_options.accelerator_options
        )

        self.device = device
        self.ocr_scale = 3

    def process_all_pages(self, url: str, input_doc, pages: List[Page]) -> List[Page]:
        """Main GPU processing pipeline with timing logs."""
        n_pages = len(pages)
        print(f"\n🚀 GPU Processing {n_pages} pages...")

        t_all_start = time.perf_counter()

        # Create ConversionResult for models
        conv_res = ConversionResult(input=input_doc)
        conv_res.pages = pages

        # Step 1: Layout Detection (batched internally)
        print("  1️⃣ Running layout detection...")
        t_layout_start = time.perf_counter()
        pages_with_layout = list(self.layout_model(conv_res, pages))
        t_layout = time.perf_counter() - t_layout_start
        print(f"     ⏱ layout: {fmt_secs(t_layout)}")

        # Step 2: Identify regions needing OCR
        print("  2️⃣ Identifying OCR regions...")
        t_ocr_detect_start = time.perf_counter()
        ocr_tasks = self._identify_ocr_regions(pages_with_layout)
        t_ocr_detect = time.perf_counter() - t_ocr_detect_start
        print(f"     Found {len(ocr_tasks)} text regions needing OCR")
        print(f"     ⏱ ocr-detection: {fmt_secs(t_ocr_detect)}")

        # Step 3: Batch OCR processing
        pages_with_ocr = pages_with_layout
        t_ocr_run = 0.0
        if ocr_tasks:
            print(f"  3️⃣ Batch processing OCR ({len(ocr_tasks)} regions)...")
            t_ocr_run_start = time.perf_counter()
            ocr_results = self._batch_ocr(ocr_tasks)
            pages_with_ocr = self._apply_ocr_results(pages_with_layout, ocr_results)
            t_ocr_run = time.perf_counter() - t_ocr_run_start
            print(f"     ⏱ ocr-generate+apply: {fmt_secs(t_ocr_run)}")
        else:
            print("  3️⃣ No OC"
                  "R needed - all text extractable")

        # Step 4: Table Structure Analysis
        t_tables = 0.0
        if self.pipeline_options.do_table_structure:
            print("  4️⃣ Analyzing table structures...")
            conv_res.pages = pages_with_ocr
            t_tables_start = time.perf_counter()
            pages_with_tables = list(self.table_model(conv_res, pages_with_ocr))
            t_tables = time.perf_counter() - t_tables_start
            print(f"     ⏱ tables: {fmt_secs(t_tables)}")
            print_timing_summary()
        else:
            pages_with_tables = pages_with_ocr

        t_all = time.perf_counter() - t_all_start
        print("✅ GPU processing complete!")
        print(
            "   ─ Timings ─ "
            f"layout: {fmt_secs(t_layout)} | "
            f"ocr-detect: {fmt_secs(t_ocr_detect)} | "
            f"ocr: {fmt_secs(t_ocr_run)} | "
            f"tables: {fmt_secs(t_tables)} | "
            f"total: {fmt_secs(t_all)}"
        )

        self.end_of_run_regression(url=url, pages_list=pages_with_tables, mode="compare")

        return pages_with_tables

    def _identify_ocr_regions(self, pages: List[Page]) -> List[dict]:
        """Find text regions without extractable text."""
        text_labels = {
            DocItemLabel.TEXT,
            DocItemLabel.SECTION_HEADER,
            DocItemLabel.CAPTION,
            DocItemLabel.FOOTNOTE,
            DocItemLabel.LIST_ITEM,
        }

        ocr_tasks = []
        for page in pages:
            if not page.predictions.layout:
                continue

            for cluster in page.predictions.layout.clusters:
                if cluster.label in text_labels and len(cluster.cells) == 0:
                    # Text region with no text - needs OCR
                    if page._backend:
                        img = page._backend.get_page_image(
                            scale=self.ocr_scale,
                            cropbox=cluster.bbox
                        )
                        ocr_tasks.append({
                            'page_no': page.page_no,
                            'cluster_id': cluster.id,
                            'bbox': cluster.bbox,
                            'image': np.array(img)
                        })

        return ocr_tasks

    def _batch_ocr(self, ocr_tasks: List[dict], batch_size: int = 32) -> List[tuple]:
        """Batch process all OCR regions."""
        results = []

        for i in range(0, len(ocr_tasks), batch_size):
            batch = ocr_tasks[i:i + batch_size]
            images = [task['image'] for task in batch]

            # Single GPU call for batch
            pixel_values = self.ocr_processor(
                images=images,
                return_tensors="pt"
            ).pixel_values.to(self.device)

            with torch.no_grad():
                generated_ids = self.ocr_model.generate(pixel_values, max_new_tokens=256)

            texts = self.ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)

            for task, text in zip(batch, texts):
                results.append((task, text))

        return results

    def _apply_ocr_results(self, pages: List[Page], ocr_results: List[tuple]) -> List[Page]:
        """Apply OCR text back to clusters."""
        results_by_page = {}
        for task, text in ocr_results:
            page_no = task['page_no']
            if page_no not in results_by_page:
                results_by_page[page_no] = []
            results_by_page[page_no].append((task, text))

        for page in pages:
            if page.page_no in results_by_page:
                for task, text in results_by_page[page.page_no]:
                    # Add OCR text to the appropriate cluster
                    for cluster in page.predictions.layout.clusters:
                        if cluster.id == task['cluster_id']:
                            # Create OCR text cell
                            ocr_cell = TextCell(
                                text=text,
                                orig=text,
                                from_ocr=True,
                                confidence=0.9,
                                rect=BoundingRectangle.from_bounding_box(task['bbox'])
                            )
                            cluster.cells.append(ocr_cell)
                            if page.parsed_page:
                                page.parsed_page.textline_cells.append(ocr_cell)
                            break

        return pages

    def cleanup(self):
        """Release GPU memory by deleting models and clearing cache."""
        print("\n🧹 Cleaning up GPU resources...")

        # Delete models
        if hasattr(self, 'layout_model'):
            if hasattr(self.layout_model, 'layout_predictor'):
                if hasattr(self.layout_model.layout_predictor, '_model'):
                    del self.layout_model.layout_predictor._model
            del self.layout_model

        if hasattr(self, 'ocr_model'):
            del self.ocr_model

        if hasattr(self, 'ocr_processor'):
            del self.ocr_processor

        if hasattr(self, 'table_model'):
            if hasattr(self.table_model, 'tf_predictor'):
                if hasattr(self.table_model.tf_predictor, '_model'):
                    del self.table_model.tf_predictor._model
            del self.table_model

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("✓ GPU memory cleared")
        elif torch.backends.mps.is_available():
            # MPS doesn't have explicit cache clearing, but we can synchronize
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
            print("✓ MPS memory released")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup on exit."""
        self.cleanup()
        return False

    @staticmethod
    def safe_id(url: str) -> str:
        # Strip protocol and non-filename chars
        return re.sub(r'[^A-Za-z0-9._-]+', '_', url)

    def end_of_run_regression(self, url, pages_list, mode: str = "compare"):
        tol = Tolerances(bbox_abs=1.0, bbox_rel=0.01, iou_min=0.98, text_case_insensitive=False)
        runner = TableRegressionRunner(
            out_dir=os.getenv("TS_REGRESSION_DIR", "./tf_regression"),
            mode=mode,
            tolerances=tol
        )
        doc_id = self.safe_id(url)  # ✅ make URL safe for filenames
        result = runner.run(doc_id, pages_list)
        return result
