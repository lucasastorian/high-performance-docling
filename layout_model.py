import copy
import logging
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from docling_core.types.doc import DocItemLabel
from PIL import Image

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import BoundingBox, Cluster, LayoutPrediction, Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_V2, LayoutModelConfig
from docling.datamodel.pipeline_options import LayoutOptions
from docling.datamodel.settings import settings
from docling.models.base_model import BasePageModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device
from docling.utils.layout_postprocessor import LayoutPostprocessor
from docling.utils.profiling import TimeRecorder
from docling.utils.visualization import draw_clusters

from layout_predictor import LayoutPredictor

_log = logging.getLogger(__name__)


class LayoutModel(BasePageModel):
    TEXT_ELEM_LABELS = [
        DocItemLabel.TEXT,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.CAPTION,
        DocItemLabel.CHECKBOX_UNSELECTED,
        DocItemLabel.CHECKBOX_SELECTED,
        DocItemLabel.SECTION_HEADER,
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.CODE,
        DocItemLabel.LIST_ITEM,
        DocItemLabel.FORMULA,
    ]
    PAGE_HEADER_LABELS = [DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER]

    TABLE_LABELS = [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]
    FIGURE_LABEL = DocItemLabel.PICTURE
    FORMULA_LABEL = DocItemLabel.FORMULA
    CONTAINER_LABELS = [DocItemLabel.FORM, DocItemLabel.KEY_VALUE_REGION]

    def __init__(
            self,
            artifacts_path: Optional[Path],
            accelerator_options: AcceleratorOptions,
            options: LayoutOptions,
    ):

        self.options = options

        device = decide_device(accelerator_options.device)
        layout_model_config = options.model_spec
        model_repo_folder = layout_model_config.model_repo_folder
        model_path = layout_model_config.model_path

        if artifacts_path is None:
            artifacts_path = (
                    self.download_models(layout_model_config=layout_model_config)
                    / model_path
            )
        else:
            if (artifacts_path / model_repo_folder).exists():
                artifacts_path = artifacts_path / model_repo_folder / model_path
            elif (artifacts_path / model_path).exists():
                warnings.warn(
                    "The usage of artifacts_path containing directly "
                    f"{model_path} is deprecated. Please point "
                    "the artifacts_path to the parent containing "
                    f"the {model_repo_folder} folder.",
                    DeprecationWarning,
                    stacklevel=3,
                )
                artifacts_path = artifacts_path / model_path

        self.layout_predictor = LayoutPredictor(
            artifact_path=str(artifacts_path),
            device=device,
            num_threads=accelerator_options.num_threads,
        )

    @staticmethod
    def download_models(
            local_dir: Optional[Path] = None,
            force: bool = False,
            progress: bool = False,
            layout_model_config: LayoutModelConfig = DOCLING_LAYOUT_V2,
    ) -> Path:
        return download_hf_model(
            repo_id=layout_model_config.repo_id,
            revision=layout_model_config.revision,
            local_dir=local_dir,
            force=force,
            progress=progress,
        )

    def draw_clusters_and_cells_side_by_side(
            self, conv_res, page, clusters, mode_prefix: str, show: bool = False
    ):
        """
        Draws a page image side by side with clusters filtered into two categories:
        - Left: Clusters excluding FORM, KEY_VALUE_REGION, and PICTURE.
        - Right: Clusters including FORM, KEY_VALUE_REGION, and PICTURE.
        Includes label names and confidence scores for each cluster.
        """
        scale_x = page.image.width / page.size.width
        scale_y = page.image.height / page.size.height

        # Filter clusters for left and right images
        exclude_labels = {
            DocItemLabel.FORM,
            DocItemLabel.KEY_VALUE_REGION,
            DocItemLabel.PICTURE,
        }
        left_clusters = [c for c in clusters if c.label not in exclude_labels]
        right_clusters = [c for c in clusters if c.label in exclude_labels]
        # Create a deep copy of the original image for both sides
        left_image = copy.deepcopy(page.image)
        right_image = copy.deepcopy(page.image)

        # Draw clusters on both images
        draw_clusters(left_image, left_clusters, scale_x, scale_y)
        draw_clusters(right_image, right_clusters, scale_x, scale_y)
        # Combine the images side by side
        combined_width = left_image.width * 2
        combined_height = left_image.height
        combined_image = Image.new("RGB", (combined_width, combined_height))
        combined_image.paste(left_image, (0, 0))
        combined_image.paste(right_image, (left_image.width, 0))
        if show:
            combined_image.show()
        else:
            out_path: Path = (
                    Path(settings.debug.debug_output_path)
                    / f"debug_{conv_res.input.file.stem}"
            )
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / f"{mode_prefix}_layout_page_{page.page_no:05}.png"
            combined_image.save(str(out_file), format="png")

    def __call__(
            self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        import time

        # Overall timer
        t_total0 = time.perf_counter()

        # 0) Prep: collect valid pages and their base images
        t_prep0 = time.perf_counter()
        pages = list(page_batch)

        valid_pages: List[Page] = []
        valid_page_images: List[Union[Image.Image, np.ndarray]] = []

        for page in pages:
            assert page._backend is not None
            if not page._backend.is_valid():
                continue

            assert page.size is not None
            page_image = page.get_image(scale=1.0)
            assert page_image is not None

            valid_pages.append(page)
            valid_page_images.append(page_image)
        t_prep1 = time.perf_counter()

        # 1) Batch predict (LayoutPredictor has its own internal timers)
        t_batch0 = time.perf_counter()
        batch_predictions: List[List[dict]] = []
        if valid_page_images:
            batch_predictions = self.layout_predictor.predict_batch(valid_page_images)  # type: ignore[attr-defined]
        t_batch1 = time.perf_counter()

        # 2) Build clusters + 3) Postprocess per page
        t_clusters_acc = 0.0
        t_postproc_acc = 0.0
        t_visual_acc = 0.0

        valid_page_idx = 0
        for page in pages:
            assert page._backend is not None
            if not page._backend.is_valid():
                # Yield invalid pages untouched
                yield page
                continue

            page_predictions = batch_predictions[valid_page_idx]
            valid_page_idx += 1

            # 2) Build raw clusters
            t_c0 = time.perf_counter()
            clusters: List[Cluster] = []
            for ix, pred_item in enumerate(page_predictions):
                label = DocItemLabel(
                    pred_item["label"].lower().replace(" ", "_").replace("-", "_")
                )  # Temporary, until docling-ibm-model uses docling-core types
                cluster = Cluster(
                    id=ix,
                    label=label,
                    confidence=pred_item["confidence"],
                    bbox=BoundingBox.model_validate(pred_item),
                    cells=[],
                )
                clusters.append(cluster)
            t_c1 = time.perf_counter()
            t_clusters_acc += (t_c1 - t_c0)

            # Optional visualization (raw)
            if settings.debug.visualize_raw_layout:
                t_v0 = time.perf_counter()
                self.draw_clusters_and_cells_side_by_side(
                    conv_res, page, clusters, mode_prefix="raw"
                )
                t_v1 = time.perf_counter()
                t_visual_acc += (t_v1 - t_v0)

            # 3) Postprocess (merging, RO heuristics, header/footer, etc.)
            t_p0 = time.perf_counter()
            processed_clusters, processed_cells = LayoutPostprocessor(
                page, clusters, self.options
            ).postprocess()
            t_p1 = time.perf_counter()
            t_postproc_acc += (t_p1 - t_p0)

            # Confidence bookkeeping (unchanged behavior)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "Mean of empty slice|invalid value encountered in scalar divide",
                    RuntimeWarning,
                    "numpy",
                )

                conv_res.confidence.pages[page.page_no].layout_score = float(
                    np.mean([c.confidence for c in processed_clusters])
                )

                conv_res.confidence.pages[page.page_no].ocr_score = float(
                    np.mean([c.confidence for c in processed_cells if c.from_ocr])
                )

            # Save final layout predictions
            page.predictions.layout = LayoutPrediction(clusters=processed_clusters)

            # Optional visualization (postprocessed)
            if settings.debug.visualize_layout:
                t_v0 = time.perf_counter()
                self.draw_clusters_and_cells_side_by_side(
                    conv_res, page, processed_clusters, mode_prefix="postprocessed"
                )
                t_v1 = time.perf_counter()
                t_visual_acc += (t_v1 - t_v0)

            # Yield the page
            yield page

        # Print super-basic timing summary for the whole layout call
        t_total1 = time.perf_counter()
        prep = t_prep1 - t_prep0
        batch = t_batch1 - t_batch0
        total = t_total1 - t_total0
        # The remainder gives you "other" overhead not in the three buckets, if any.
        other = total - (prep + batch + t_clusters_acc + t_postproc_acc + t_visual_acc)

        print(
            "[layout] total: "
            f"{total:.3f}s | prep_pages: {prep:.3f}s | "
            f"batch_predict: {batch:.3f}s | cluster_build: {t_clusters_acc:.3f}s | "
            f"postprocess: {t_postproc_acc:.3f}s | visualize: {t_visual_acc:.3f}s | "
            f"other: {other:.3f}s"
        )
