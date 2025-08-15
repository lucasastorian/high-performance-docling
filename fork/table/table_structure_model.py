import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Any, Dict, List
from PIL import ImageDraw

from docling_core.types.doc import BoundingBox, DocItemLabel, TableCell
from docling_core.types.doc.page import BoundingRectangle, TextCellUnit

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import TableFormerMode, TableStructureOptions
from docling.datamodel.settings import settings
from docling.models.base_model import BasePageModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

from base_models import Table, TableStructurePrediction, Page

from fork.table.tf_predictor import TFPredictor
from fork.timers import _CPUTimer, _CudaTimer


class TableStructureModel(BasePageModel):
    _model_repo_folder = "ds4sd--docling-models"
    _model_path = "model_artifacts/tableformer"

    def __init__(
            self,
            enabled: bool,
            artifacts_path: Optional[Path],
            options: TableStructureOptions,
            accelerator_options: AcceleratorOptions,
    ):
        self.options = options
        self.do_cell_matching = self.options.do_cell_matching
        self.mode = self.options.mode

        self.enabled = enabled
        if self.enabled:
            if artifacts_path is None:
                artifacts_path = self.download_models() / self._model_path
            else:
                # will become the default in the future
                if (artifacts_path / self._model_repo_folder).exists():
                    artifacts_path = (
                            artifacts_path / self._model_repo_folder / self._model_path
                    )
                elif (artifacts_path / self._model_path).exists():
                    warnings.warn(
                        "The usage of artifacts_path containing directly "
                        f"{self._model_path} is deprecated. Please point "
                        "the artifacts_path to the parent containing "
                        f"the {self._model_repo_folder} folder.",
                        DeprecationWarning,
                        stacklevel=3,
                    )
                    artifacts_path = artifacts_path / self._model_path

            if self.mode == TableFormerMode.ACCURATE:
                artifacts_path = artifacts_path / "accurate"
            else:
                artifacts_path = artifacts_path / "fast"

            # Third Party
            import docling_ibm_models.tableformer.common as c

            device = decide_device(accelerator_options.device)

            # Keep original behavior: disable MPS to avoid perf/num issues on Apple Silicon
            if device == AcceleratorDevice.MPS.value:
                device = AcceleratorDevice.CPU.value

            self.tm_config = c.read_config(f"{artifacts_path}/tm_config.json")
            self.tm_config["model"]["save_dir"] = artifacts_path
            self.tm_model_type = self.tm_config["model"]["type"]

            self.tf_predictor = TFPredictor(
                self.tm_config, device, accelerator_options.num_threads
            )
            self.scale = 2.0  # Scale up table input images to ~144 dpi

    @staticmethod
    def download_models(
            local_dir: Optional[Path] = None, force: bool = False, progress: bool = False
    ) -> Path:
        return download_hf_model(
            repo_id="ds4sd/docling-models",
            revision="v2.2.0",
            local_dir=local_dir,
            force=force,
            progress=progress,
        )

    def draw_table_and_cells(
            self,
            conv_res: ConversionResult,
            page: Page,
            tbl_list: Iterable[Table],
            show: bool = False,
    ):
        assert page._backend is not None
        assert page.size is not None

        image = page._backend.get_page_image()  # avoid drawing on saved ones
        scale_x = image.width / page.size.width
        scale_y = image.height / page.size.height

        draw = ImageDraw.Draw(image)

        for table_element in tbl_list:
            x0, y0, x1, y1 = table_element.cluster.bbox.as_tuple()
            y0 *= scale_y
            y1 *= scale_y
            x0 *= scale_x
            x1 *= scale_x

            draw.rectangle([(x0, y0), (x1, y1)], outline="red")

            for cell in table_element.cluster.cells:
                x0, y0, x1, y1 = cell.rect.to_bounding_box().as_tuple()
                x0 *= scale_x
                x1 *= scale_x
                y0 *= scale_y
                y1 *= scale_y
                draw.rectangle([(x0, y0), (x1, y1)], outline="green")

            for tc in table_element.table_cells:
                if tc.bbox is not None:
                    x0, y0, x1, y1 = tc.bbox.as_tuple()
                    x0 *= scale_x
                    x1 *= scale_x
                    y0 *= scale_y
                    y1 *= scale_y

                    width = 3 if tc.column_header else 1
                    draw.rectangle([(x0, y0), (x1, y1)], outline="blue", width=width)
                    draw.text(
                        (x0 + 3, y0 + 3),
                        text=f"{tc.start_row_offset_idx}, {tc.start_col_offset_idx}",
                        fill="black",
                    )
        if show:
            image.show()
        else:
            out_path: Path = (
                    Path(settings.debug.debug_output_path)
                    / f"debug_{conv_res.input.file.stem}"
            )
            out_path.mkdir(parents=True, exist_ok=True)
            out_file = out_path / f"table_struct_page_{page.page_no:05}.png"
            image.save(str(out_file), format="png")

    def __call__(
            self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        if not self.enabled:
            # passthrough
            yield from page_batch
            return

        pages_list: List[Page] = list(page_batch)

        # Batch accumulators
        page_inputs: List[dict] = []
        table_bboxes_list: List[List[List[float]]] = []  # per-page list of bboxes
        page_clusters_list: List[List[Any]] = []  # per-page list of clusters
        batched_page_indexes: List[int] = []  # map batch idx -> pages_list idx

        # Prepare pages (aggregate tokens per page; dedup by token id)
        for page_idx, page in enumerate(pages_list):
            if page._backend is None or not page._backend.is_valid():
                continue

            with TimeRecorder(conv_res, "table_structure_prep"):
                assert page.predictions.layout is not None
                assert page.size is not None

                # Always initialize predictions (like original)
                page.predictions.tablestructure = TableStructurePrediction()

                in_tables = self._get_tables_from_page(page)
                if not in_tables:
                    # Nothing to predict on this page; it will be yielded unchanged
                    continue

                # Build page_input
                # Use cached NumPy image; HWC uint8 as expected by predictor
                page_input = {
                    "width": page.size.width * self.scale,
                    "height": page.size.height * self.scale,
                    "image": page.get_image_np(scale=self.scale),
                    "index": page.word_index
                }

                # Aggregate tokens once per page when matching is on
                page_table_bboxes: List[List[float]] = []
                page_clusters: List[Any] = []

                if self.do_cell_matching:
                    seen_ids = set()
                    aggregated_tokens: List[dict] = []
                    for table_cluster, tbl_box in in_tables:
                        toks = self._get_table_tokens(page, table_cluster)
                        for tok in toks:
                            tid = tok.get("id")
                            if tid is None or tid in seen_ids:
                                continue
                            seen_ids.add(tid)
                            aggregated_tokens.append(tok)
                        page_table_bboxes.append(tbl_box)
                        page_clusters.append(table_cluster)
                    page_input["tokens"] = aggregated_tokens
                else:
                    # No tokens needed; just bboxes/clusters
                    for table_cluster, tbl_box in in_tables:
                        page_table_bboxes.append(tbl_box)
                        page_clusters.append(table_cluster)

                page_inputs.append(page_input)
                table_bboxes_list.append(page_table_bboxes)
                page_clusters_list.append(page_clusters)
                batched_page_indexes.append(page_idx)

        # If no pages required prediction, just yield originals
        if not page_inputs:
            yield from pages_list
            return

        # Create timer for detailed table structure timing
        device_type = getattr(self.tf_predictor, '_device', 'cpu')
        timer = _CudaTimer() if device_type == 'cuda' else _CPUTimer()

        # Predictor call over the whole batch (order-preserving; predictor does not reorder)
        with TimeRecorder(conv_res, "table_structure_predict"):
            with timer.time_section('tf_predictor_call'):
                all_outputs = self.tf_predictor.multi_table_predict(
                    page_inputs,
                    table_bboxes_list,
                    do_matching=self.do_cell_matching,
                    # IMPORTANT: do not pass additional flags; keep original semantics
                )

        # Map outputs back to pages/tables in strict order
        with timer.time_section('output_processing'):
            result_idx = 0
            for i, page_batch_idx in enumerate(batched_page_indexes):
                page = pages_list[page_batch_idx]
                clusters = page_clusters_list[i]
                n_tables = len(clusters)

                page_outputs = all_outputs[result_idx: result_idx + n_tables]
                result_idx += n_tables

                for output, table_cluster in zip(page_outputs, clusters):
                    table = self._process_table_output(page, table_cluster, output)
                    page.predictions.tablestructure.table_map[table_cluster.id] = table

        # Finalize timing and log results
        timer.finalize()
        print(f"   └─ table structure breakdown: tf_call={timer.get_time('tf_predictor_call'):.1f}ms "
              f"output_proc={timer.get_time('output_processing'):.1f}ms")

        # Optional debug viz; unchanged
        if settings.debug.visualize_tables:
            for page in pages_list:
                ts = getattr(page.predictions, "tablestructure", None)
                if ts and ts.table_map:
                    self.draw_table_and_cells(
                        conv_res,
                        page,
                        ts.table_map.values(),
                    )

        # Preserve original order
        for page in pages_list:
            yield page

    # ------------------------
    # Helpers (behavior-parity)
    # ------------------------

    def _get_tables_from_page(self, page: Page):
        """Return list of (cluster, scaled_bbox) for table-like clusters."""
        scl = self.scale
        return [
            (
                cluster,
                [
                    round(cluster.bbox.l) * scl,
                    round(cluster.bbox.t) * scl,
                    round(cluster.bbox.r) * scl,
                    round(cluster.bbox.b) * scl,
                ],
            )
            for cluster in page.predictions.layout.clusters
            if cluster.label in (DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX)
        ]

    def _get_table_tokens(self, page: Page, table_cluster, ios: float = 0.8):
        """
        Token aggregation via Page.word_index (TOP-LEFT origin).
        Falls back to original backend query if index is missing.
        """
        wi = getattr(page, "word_index", None)
        if wi is not None:
            # Convert cluster bbox to TOP-LEFT origin in the same page space as the index
            bbox = table_cluster.bbox.to_top_left_origin(wi.H)
            return wi.query_bbox(bbox.l, bbox.t, bbox.r, bbox.b, ios=ios, scale=self.scale)

        # Fallback: original path (kept for safety; no perf promises)
        sp = page.parsed_page
        if sp is not None:
            tcells = sp.get_cells_in_bbox(
                cell_unit=TextCellUnit.WORD,
                bbox=table_cluster.bbox,
            )
            if len(tcells) == 0:
                tcells = table_cluster.cells
        else:
            tcells = table_cluster.cells

        tokens = []
        sx = sy = self.scale
        for c in tcells:
            text = c.text.strip()
            if not text:
                continue
            bb = c.rect.to_bounding_box()
            tokens.append(
                {
                    "id": c.index,
                    "text": text,
                    "bbox": {
                        "l": bb.l * sx, "t": bb.t * sy,
                        "r": bb.r * sx, "b": bb.b * sy
                    },
                }
            )
        return tokens

    def _process_table_output(self, page: Page, table_cluster: Any, table_out: Dict) -> Table:
        """
        Convert predictor output to Table while preserving original semantics:
        - When not matching, attach text via backend.get_text_in_rect
        - Always rescale bbox back to page coords (1/self.scale)
        """
        table_cells = []

        # Original behavior: always attach text when not matching
        attach_text = not self.do_cell_matching

        tf_responses = table_out.get("tf_responses", ())
        _BoundingBox_validate = BoundingBox.model_validate
        _TableCell_validate = TableCell.model_validate
        _scale = 1.0 / self.scale
        _backend = page._backend
        _get_text = _backend.get_text_in_rect if (_backend is not None) else None

        for element in tf_responses:
            if attach_text and _get_text is not None:
                bb = _BoundingBox_validate(element["bbox"]).scaled(_scale)
                element["bbox"]["token"] = _get_text(bb)

            tc = _TableCell_validate(element)
            if tc.bbox is not None:
                tc.bbox = tc.bbox.scaled(_scale)
            table_cells.append(tc)

        pd = table_out.get("predict_details", {})
        num_rows = pd.get("num_rows", 0)
        num_cols = pd.get("num_cols", 0)
        otsl_seq = pd.get("prediction", {}).get("rs_seq", [])

        return Table(
            otsl_seq=otsl_seq,
            table_cells=table_cells,
            num_rows=num_rows,
            num_cols=num_cols,
            id=table_cluster.id,
            page_no=page.page_no,
            cluster=table_cluster,
            label=table_cluster.label,
        )
