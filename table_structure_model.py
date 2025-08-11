import warnings
import numpy as np
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Any, Dict

from docling_core.types.doc import BoundingBox, DocItemLabel, TableCell
from docling_core.types.doc.page import (
    BoundingRectangle,
    TextCellUnit,
)
from PIL import ImageDraw

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Table, TableStructurePrediction
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    TableFormerMode,
    TableStructureOptions,
)
from docling.datamodel.settings import settings
from docling.models.base_model import BasePageModel
from docling.models.utils.hf_model_download import download_hf_model
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder

from page_model import Page
from tf_predictor import TFPredictor
from table_timing_debug import get_timing_collector


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
            print(f"Running Table predictions on {device}")

            self.tm_config = c.read_config(f"{artifacts_path}/tm_config.json")
            self.tm_config["model"]["save_dir"] = artifacts_path
            self.tm_model_type = self.tm_config["model"]["type"]

            self.tf_predictor = TFPredictor(
                self.tm_config, device, accelerator_options.num_threads
            )

            # DISABLED: Baseline caching was taking 1.6s!
            # self.tf_predictor.enable_baseline_cache("/users/lucasastorian/docling-ibm-models/baseline_cache/", mode="baseline")
            self.scale = 2.0  # Scale up table input images to 144 dpi

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

        image = (
            page._backend.get_page_image()
        )  # make new image to avoid drawing on the saved ones

        scale_x = image.width / page.size.width
        scale_y = image.height / page.size.height

        draw = ImageDraw.Draw(image)

        for table_element in tbl_list:
            x0, y0, x1, y1 = table_element.cluster.bbox.as_tuple()
            y0 *= scale_x
            y1 *= scale_y
            x0 *= scale_x
            x1 *= scale_x

            draw.rectangle([(x0, y0), (x1, y1)], outline="red")

            for cell in table_element.cluster.cells:
                x0, y0, x1, y1 = cell.rect.to_bounding_box().as_tuple()
                x0 *= scale_x
                x1 *= scale_x
                y0 *= scale_x
                y1 *= scale_y

                draw.rectangle([(x0, y0), (x1, y1)], outline="green")

            for tc in table_element.table_cells:
                if tc.bbox is not None:
                    x0, y0, x1, y1 = tc.bbox.as_tuple()
                    x0 *= scale_x
                    x1 *= scale_x
                    y0 *= scale_x
                    y1 *= scale_y

                    if tc.column_header:
                        width = 3
                    else:
                        width = 1
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

    def __call__(self, conv_res: ConversionResult, page_batch: Iterable[Page]) -> Iterable[Page]:
        import time
        from table_timing_debug import get_timing_collector

        t_full_start = time.perf_counter()
        timer = get_timing_collector()

        if not self.enabled:
            yield from page_batch
            return

        pages_list = list(page_batch)

        # Per-batch accumulators
        page_inputs: list[dict] = []
        table_bboxes_list: list[list[list[float]]] = []  # one bbox list per page
        page_clusters_list: list[list[Any]] = []  # clusters for each page, same order as bboxes
        batched_page_indexes: list[int] = []  # map batch idx -> original pages_list idx

        # Prepare per-page structures (and create empty predictions so downstream code doesn't explode)
        timer.start("prep_inputs")
        for page_idx, page in enumerate(pages_list):
            assert page is not None
            if page._backend is None or not page._backend.is_valid():
                # Invalid page: keep it untouched, we won't include it in the batch
                continue

            with TimeRecorder(conv_res, "table_structure_prep"):
                assert page.predictions.layout is not None
                assert page.size is not None

                # Always initialize, even if no tables. Matches original behavior.
                page.predictions.tablestructure = TableStructurePrediction()

                in_tables = self._get_tables_from_page(page)
                if not in_tables:
                    # Nothing to predict for this page; skip batching but still yield later.
                    continue

                # Only aggregate tokens if we're doing cell matching
                page_table_bboxes = []
                page_clusters = []

                if self.do_cell_matching:
                    timer.start("prep_inputs:gather_tokens")
                    seen_ids = set()
                    aggregated_tokens = []

                    for table_cluster, tbl_box in in_tables:
                        table_tokens = self._get_table_tokens(page, table_cluster)
                        for tok in table_tokens:
                            tid = tok.get("id")
                            if tid is None or tid in seen_ids:
                                continue
                            seen_ids.add(tid)
                            aggregated_tokens.append(tok)

                        page_table_bboxes.append(tbl_box)
                        page_clusters.append(table_cluster)
                    timer.end("prep_inputs:gather_tokens")
                else:
                    # No tokens needed - just collect bboxes
                    for table_cluster, tbl_box in in_tables:
                        page_table_bboxes.append(tbl_box)
                        page_clusters.append(table_cluster)

                # Build the page_input
                timer.start("prep_inputs:image_np")
                page_input = {
                    "width": page.size.width * self.scale,
                    "height": page.size.height * self.scale,
                    # NOTE: Numpy image at scale 2 is cached during Lambda preprocessing, so getting image is instantaneous.
                    "image": page.get_image_np(scale=self.scale),
                }
                timer.end("prep_inputs:image_np")

                # Only add tokens if matching
                if self.do_cell_matching:
                    page_input["tokens"] = aggregated_tokens

                page_inputs.append(page_input)
                table_bboxes_list.append(page_table_bboxes)
                page_clusters_list.append(page_clusters)
                batched_page_indexes.append(page_idx)

        timer.end("prep_inputs")

        # If there is nothing to run, just yield the originals
        if not page_inputs:
            yield from pages_list
            return

        # Run the batched predictor
        import time
        t_predict_start = time.perf_counter()
        with TimeRecorder(conv_res, "table_structure_predict"):
            all_outputs = self.tf_predictor.multi_table_predict(
                page_inputs,
                table_bboxes_list,
                do_matching=self.do_cell_matching,
                correct_overlapping_cells=False,
                sort_row_col_indexes=True,
                doc_id="document",
                start_page_idx=0,
            )
        t_predict_end = time.perf_counter()
        print(f"⏱ multi_table_predict took: {t_predict_end - t_predict_start:.3f}s")

        # Map flat outputs back to pages and clusters in strict order
        timer.start("assign_outputs")
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

        timer.end("assign_outputs")

        # Optional debug viz
        if settings.debug.visualize_tables:
            for page in pages_list:
                if (
                        getattr(page.predictions, "tablestructure", None)
                        and page.predictions.tablestructure.table_map
                ):
                    self.draw_table_and_cells(
                        conv_res,
                        page,
                        page.predictions.tablestructure.table_map.values(),
                    )

        # Preserve original order
        for page in pages_list:
            yield page

        t_full_end = time.perf_counter()
        print(f"⏱ TableStructureModel.__call__ TOTAL: {t_full_end - t_full_start:.3f}s")

    def _get_tables_from_page(self, page: Page):
        """Returns all the tables on a page"""
        return [
            (
                cluster,
                [
                    round(cluster.bbox.l) * self.scale,
                    round(cluster.bbox.t) * self.scale,
                    round(cluster.bbox.r) * self.scale,
                    round(cluster.bbox.b) * self.scale,
                ],
            )
            for cluster in page.predictions.layout.clusters
            if cluster.label
               in [DocItemLabel.TABLE, DocItemLabel.DOCUMENT_INDEX]
        ]

    def _get_table_tokens(self, page, table_cluster, ios=0.8):
        bbox = table_cluster.bbox.to_top_left_origin(page.word_index.page_height)
        return page.word_index.query_bbox(bbox.l, bbox.t, bbox.r, bbox.b, ios=ios, scale=self.scale)

        # # Ensure tokens are prebuilt
        # toks = page.tokens_np
        # if toks is None or toks.size == 0:
        #     return []
        #
        # # Table bbox → TOP-LEFT origin (unscaled)
        # H = float(page.size.height)
        # tbl_tl = table_cluster.bbox.to_top_left_origin(page_height=H)
        # lb, tb, rb, bb = tbl_tl.l, tbl_tl.t, tbl_tl.r, tbl_tl.b
        #
        # # Vectorized IOS filter
        # L, T, R, B = toks['l'], toks['t'], toks['r'], toks['b']
        # inter_w = np.maximum(0.0, np.minimum(R, rb) - np.maximum(L, lb))
        # inter_h = np.maximum(0.0, np.minimum(B, bb) - np.maximum(T, tb))
        # inter = inter_w * inter_h
        # area = (R - L) * (B - T)
        # keep = inter >= (ios * area)
        #
        # if not np.any(keep):
        #     return []
        #
        # idx = np.nonzero(keep)[0]
        # sel = toks[idx]
        #
        # # Package for TF: scale now (cheap, one pass)
        # s = float(self.scale)
        # return [
        #     {
        #         "id": int(t['id']),
        #         "text": "",  # keep light; attach later only if needed
        #         "bbox": {
        #             "l": float(t['l'] * s),
        #             "t": float(t['t'] * s),
        #             "r": float(t['r'] * s),
        #             "b": float(t['b'] * s),
        #         },
        #     }
        #     for t in sel
        # ]
        #
        # sp = page.parsed_page
        # if sp is not None:
        #     tcells = sp.get_cells_in_bbox(
        #         cell_unit=TextCellUnit.WORD,
        #         bbox=table_cluster.bbox,
        #     )
        #     if len(tcells) == 0:
        #         # In case word-level cells yield empty
        #         tcells = table_cluster.cells
        # else:
        #     # Otherwise - we use normal (line/phrase) cells
        #     tcells = table_cluster.cells
        #
        # tokens = []
        # sx = sy = self.scale  # Pre-compute scale factors
        #
        # for c in tcells:
        #     # Only allow non empty strings (spaces) into the cells of a table
        #     text = c.text.strip()
        #     if not text:
        #         continue
        #
        #     # Direct bbox calculation without deep copy or intermediate objects
        #     bb = c.rect.to_bounding_box()
        #     tokens.append(
        #         {
        #             "id": c.index,
        #             "text": text,
        #             "bbox": {
        #                 "l": bb.l * sx,
        #                 "t": bb.t * sy,
        #                 "r": bb.r * sx,
        #                 "b": bb.b * sy,
        #             },
        #         }
        #     )
        #
        # return tokens

    # def _process_table_output(self, page: Page, table_cluster: Any, table_out: Dict) -> Table:
    #     timer = get_timing_collector()
    #     table_cells = []
    #
    #     # Option to attach text when not matching (default: True for compatibility)
    #     attach_text = not self.do_cell_matching and getattr(self.options, 'attach_cell_text', True)
    #
    #     with timer.scoped("assign_outputs:validate_cells"):
    #         for element in table_out["tf_responses"]:
    #             if attach_text:
    #                 # When not matching, extract text for each cell
    #                 with timer.scoped("assign_outputs:extract_text"):
    #                     the_bbox = BoundingBox.model_validate(
    #                         element["bbox"]
    #                     ).scaled(1 / self.scale)
    #                     text_piece = page._backend.get_text_in_rect(
    #                         the_bbox
    #                     )
    #                     element["bbox"]["token"] = text_piece
    #
    #             # Use model_validate to ensure the validator runs and text field is populated
    #             tc = TableCell.model_validate(element)
    #             if tc.bbox is not None:
    #                 tc.bbox = tc.bbox.scaled(1 / self.scale)
    #             table_cells.append(tc)
    #
    #     assert "predict_details" in table_out
    #
    #     # Retrieving cols/rows, after post processing:
    #     num_rows = table_out["predict_details"].get("num_rows", 0)
    #     num_cols = table_out["predict_details"].get("num_cols", 0)
    #     otsl_seq = (
    #         table_out["predict_details"]
    #         .get("prediction", {})
    #         .get("rs_seq", [])
    #     )
    #
    #     return Table(
    #         otsl_seq=otsl_seq,
    #         table_cells=table_cells,
    #         num_rows=num_rows,
    #         num_cols=num_cols,
    #         id=table_cluster.id,
    #         page_no=page.page_no,
    #         cluster=table_cluster,
    #         label=table_cluster.label,
    #     )

    def _process_table_output(self, page: Page, table_cluster: Any, table_out: Dict) -> Table:
        timer = get_timing_collector()
        table_cells = []

        # Reduce overhead of per-element timing; set to 2/4/8 to sample
        SAMPLE_EVERY = 1
        attach_text = not self.do_cell_matching and getattr(self.options, 'attach_cell_text', True)

        with timer.scoped("assign_outputs"):  # per-table envelope
            tf_responses = table_out.get("tf_responses", ())
            _BoundingBox_validate = BoundingBox.model_validate
            _TableCell_validate = TableCell.model_validate
            _scale = 1.0 / self.scale
            _backend = page._backend
            _get_text = _backend.get_text_in_rect if (_backend is not None) else None

            # Hot loop: split into coarse buckets so we can see where time goes
            with timer.scoped("assign_outputs:loop"):
                for idx, element in enumerate(tf_responses):
                    do_sample = (idx % SAMPLE_EVERY) == 0

                    # 1) Optional text extraction (usually dominant when enabled)
                    if attach_text and _get_text is not None:
                        if do_sample:
                            t_ctx = timer.scoped("assign_outputs:extract_text")
                            t_ctx.__enter__()
                        bb = _BoundingBox_validate(element["bbox"]).scaled(_scale)
                        text_piece = _get_text(bb)
                        element["bbox"]["token"] = text_piece
                        if do_sample:
                            t_ctx.__exit__(None, None, None)

                    # 2) Pydantic validation
                    if do_sample:
                        t_ctx = timer.scoped("assign_outputs:validate_cell")
                        t_ctx.__enter__()
                    tc = _TableCell_validate(element)
                    if do_sample:
                        t_ctx.__exit__(None, None, None)

                    # 3) Bbox rescale back to page coords
                    if tc.bbox is not None:
                        if do_sample:
                            t_ctx = timer.scoped("assign_outputs:scale_bbox")
                            t_ctx.__enter__()
                        tc.bbox = tc.bbox.scaled(_scale)
                        if do_sample:
                            t_ctx.__exit__(None, None, None)

                    table_cells.append(tc)

            # 4) Predict-details parse (should be tiny; measured separately)
            with timer.scoped("assign_outputs:predict_details_parse"):
                pd = table_out.get("predict_details", {})
                num_rows = pd.get("num_rows", 0)
                num_cols = pd.get("num_cols", 0)
                otsl_seq = pd.get("prediction", {}).get("rs_seq", [])

            # 5) Construct the Table object (should be near-zero)
            with timer.scoped("assign_outputs:table_ctor"):
                result = Table(
                    otsl_seq=otsl_seq,
                    table_cells=table_cells,
                    num_rows=num_rows,
                    num_cols=num_cols,
                    id=table_cluster.id,
                    page_no=page.page_no,
                    cluster=table_cluster,
                    label=table_cluster.label,
                )

        return result
