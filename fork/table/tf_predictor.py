#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import glob
import json
import logging
import os
import threading
from itertools import groupby
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_model

import docling_ibm_models.tableformer.common as c
import docling_ibm_models.tableformer.data_management.transforms as T
import docling_ibm_models.tableformer.settings as s
import docling_ibm_models.tableformer.utils.utils as u
from docling_ibm_models.tableformer.otsl import otsl_to_html
from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler

from fork.table.tablemodel04_rs import TableModel04_rs
from fork.table.matching_post_processor import MatchingPostProcessor
from fork.timers import _CPUTimer, _CudaTimer
from fork.table.tf_cell_matcher import CellMatcher

LOG_LEVEL = logging.WARN

logger = s.get_custom_logger(__name__, LOG_LEVEL)

_model_init_lock = threading.Lock()


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def otsl_sqr_chk(rs_list, logdebug):
    rs_list_split = [
        list(group) for k, group in groupby(rs_list, lambda x: x == "nl") if not k
    ]
    isSquare = True
    if len(rs_list_split) > 0:
        init_tag_len = len(rs_list_split[0]) + 1

        totcelnum = rs_list.count("fcel") + rs_list.count("ecel")
        if logdebug:
            logger.debug("Total number of cells = {}".format(totcelnum))

        for ind, ln in enumerate(rs_list_split):
            ln.append("nl")
            if logdebug:
                logger.debug("{}".format(ln))
            if len(ln) != init_tag_len:
                isSquare = False
        if isSquare:
            if logdebug:
                logger.debug(
                    "{}*OK* Table is square! *OK*{}".format(
                        bcolors.OKGREEN, bcolors.ENDC
                    )
                )
        else:
            if logdebug:
                err_name = "{}***** ERR ******{}"
                logger.debug(err_name.format(bcolors.FAIL, bcolors.ENDC))
                logger.debug(
                    "{}*ERR* Table is not square! *ERR*{}".format(
                        bcolors.FAIL, bcolors.ENDC
                    )
                )
    return isSquare


class TFPredictor:
    r"""
    Table predictions for the in-memory Docling API
    """

    def __init__(self, config, device: str = "cpu", num_threads: int = 4):
        r"""
        Parameters
        ----------
        config : dict Parameters configuration
        device: (Optional) torch device to run the inference.
        num_threads: (Optional) Number of threads to run the inference if device = 'cpu'

        Raises
        ------
        ValueError
        When the model cannot be found
        """
        # self._device = torch.device(device)
        self._device = device
        self._log().info("Running on device: {}".format(device))

        self._config = config
        self.enable_post_process = True

        self._padding = config["predict"].get("padding", False)
        self._padding_size = config["predict"].get("padding_size", 10)

        self._cell_matcher = CellMatcher(config)
        self._post_processor = MatchingPostProcessor(config)

        self._init_word_map()

        # Set the number of threads
        if device == "cpu":
            self._num_threads = num_threads
            torch.set_num_threads(self._num_threads)

        # Load the model
        self._model = self._load_model()
        self._model.eval()
        self._prof = config["predict"].get("profiling", False)
        self._profiling_agg_window = config["predict"].get("profiling_agg_window", None)
        if self._profiling_agg_window is not None:
            AggProfiler(self._profiling_agg_window)
        else:
            AggProfiler()

    def _init_word_map(self):
        self._prepared_data_dir = c.safe_get_parameter(
            self._config, ["dataset", "prepared_data_dir"], required=False
        )

        if self._prepared_data_dir is None:
            self._word_map = c.safe_get_parameter(
                self._config, ["dataset_wordmap"], required=True
            )
        else:
            data_name = c.safe_get_parameter(
                self._config, ["dataset", "name"], required=True
            )
            word_map_fn = c.get_prepared_data_filename("WORDMAP", data_name)

            # Load word_map
            with open(os.path.join(self._prepared_data_dir, word_map_fn), "r") as f:
                self._log().debug("Load WORDMAP from: {}".format(word_map_fn))
                self._word_map = json.load(f)

        self._init_data = {"word_map": self._word_map}
        # Prepare a reversed index for the word map
        self._rev_word_map = {v: k for k, v in self._word_map["word_map_tag"].items()}

    def get_init_data(self):
        r"""
        Return the initialization data
        """
        return self._init_data

    def get_model(self):
        r"""
        Return the loaded model
        """
        return self._model

    def _load_model(self):
        r"""
        Load the proper model
        """

        self._model_type = self._config["model"]["type"]

        # Use lock to prevent threading issues during model initialization
        with _model_init_lock:
            model = TableModel04_rs(self._config, self._init_data, self._device)

            self._remove_padding = False
            if self._model_type == "TableModel02":
                self._remove_padding = True

            # Load model from safetensors
            save_dir = self._config["model"]["save_dir"]
            models_fn = glob.glob(f"{save_dir}/tableformer_*.safetensors")
            if not models_fn:
                err_msg = "Not able to find a model file for {}".format(
                    self._model_type
                )
                self._log().error(err_msg)
                raise ValueError(err_msg)
            model_fn = models_fn[
                0
            ]  # Take the first tableformer safetensors file inside the save_dir
            missing, unexpected = load_model(model, model_fn, device=self._device)
            if missing or unexpected:
                err_msg = "Not able to load the model weights for {}".format(
                    self._model_type
                )
                self._log().error(err_msg)
                raise ValueError(err_msg)

            # Setup model for inference (bf16 conversion, eval mode) AFTER loading weights
            model.setup_for_inference()
            self._log().info("Model configured for optimized inference (bf16 transformers, fp32 encoder)")

            # Prepare the encoder for inference AFTER loading weights
            # This handles fusion, compilation, and graph capture in correct order
            if hasattr(model, '_encoder') and hasattr(model._encoder, 'prepare_for_inference'):
                # prepare_for_inference returns the encoder (may be compiled wrapper)
                model._encoder = model._encoder.prepare_for_inference(device=self._device)
                self._log().info("Encoder prepared for inference after weight loading")

        return model

    def get_device(self):
        return self._device

    def get_model_type(self):
        return self._model_type

    def _log(self):
        # Setup a custom logger
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    def _deletebbox(self, listofbboxes, index):
        newlist = []
        for i in range(len(listofbboxes)):
            bbox = listofbboxes[i]
            if i not in index:
                newlist.append(bbox)
        return newlist

    def _remove_bbox_span_desync(self, prediction):
        # Delete 1 extra bbox after span tag
        index_to_delete_from = 0
        indexes_to_delete = []
        newbboxes = []
        for html_elem in prediction["html_seq"]:
            if html_elem == "<td>":
                index_to_delete_from += 1
            if html_elem == ">":
                index_to_delete_from += 1
                # remove element from bboxes
                self._log().debug(
                    "========= DELETE BBOX INDEX: {}".format(index_to_delete_from)
                )
                indexes_to_delete.append(index_to_delete_from)

        newbboxes = self._deletebbox(prediction["bboxes"], indexes_to_delete)
        return newbboxes

    def _check_bbox_sync(self, prediction):
        bboxes = []
        match = False
        # count bboxes
        count_bbox = len(prediction["bboxes"])
        # count td tags
        count_td = 0
        for html_elem in prediction["html_seq"]:
            if html_elem == "<td>" or html_elem == ">":
                count_td += 1
            if html_elem in ["fcel", "ecel", "ched", "rhed", "srow"]:
                count_td += 1
        self._log().debug(
            "======================= PREDICTED BBOXES: {}".format(count_bbox)
        )
        self._log().debug(
            "=======================  PREDICTED CELLS: {}".format(count_td)
        )
        if count_bbox != count_td:
            bboxes = self._remove_bbox_span_desync(prediction)
        else:
            bboxes = prediction["bboxes"]
            match = True
        return match, bboxes

    def page_coords_to_table_coords(self, bbox, table_bbox, im_width, im_height):
        r"""
        Transforms given bbox from page coordinate system into table image coordinate system

        Parameters
        ----------
        bbox : list
            bbox to transform in page coordinates
        table_bbox : list
            table bbox, in page coordinates
        im_width : integer
            width of an image with rendered table (in pixels)
        im_height : integer
            height of an image height rendered table (in pixels)

        Returns
        -------
        bbox: list
            bbox with transformed coordinates
        """
        # Coordinates of given bbox
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]

        # Coordinates of table bbox
        t_x1 = table_bbox[0]
        t_y1 = table_bbox[1]
        t_x2 = table_bbox[2]
        t_y2 = table_bbox[3]

        # Table width / height
        tw = t_x2 - t_x1
        th = t_y2 - t_y1
        new_bbox = [0, 0, 0, 0]
        # Flip corners, substract table coordinates and rescale to new image size
        new_bbox[0] = im_width * (x1 - t_x1) / tw
        new_bbox[1] = im_height * (t_y2 - y2) / th
        new_bbox[2] = im_width * (x2 - t_x1) / tw
        new_bbox[3] = im_height * (t_y2 - y1) / th

        return new_bbox

    def _depad_bboxes(self, bboxes, new_image_ratio):
        r"""
        Removes padding from predicted bboxes for previously padded image

        Parameters
        ----------
        bboxes : list of lists
            list of bboxes that have to be recalculated to remove implied padding
        new_image_ratio : float
            Ratio of padded image size to the original image size

        Returns
        -------
        new_bboxes: list
            bboxes with transformed coordinates
        """
        new_bboxes = []
        c_x = 0.5
        c_y = 0.5

        self._log().debug("PREDICTED BBOXES: {}".format(bboxes))
        self._log().debug("new_image_ratio: {}".format(new_image_ratio))

        for bbox in bboxes:
            # 1. corner coords -> center coords
            cb_x1 = bbox[0] - c_x
            cb_y1 = bbox[1] - c_y
            cb_x2 = bbox[2] - c_x
            cb_y2 = bbox[3] - c_y

            # 2. center coords * new_image_ratio
            r_cb_x1 = cb_x1 * new_image_ratio
            r_cb_y1 = cb_y1 * new_image_ratio
            r_cb_x2 = cb_x2 * new_image_ratio
            r_cb_y2 = cb_y2 * new_image_ratio

            # 3. center coords -> corner coords
            x1 = r_cb_x1 + c_x
            y1 = r_cb_y1 + c_y
            x2 = r_cb_x2 + c_x
            y2 = r_cb_y2 + c_y

            x1 = np.clip(x1, 0.0, 1.0)
            y1 = np.clip(y1, 0.0, 1.0)
            x2 = np.clip(x2, 0.0, 1.0)
            y2 = np.clip(y2, 0.0, 1.0)

            new_bbox = [x1, y1, x2, y2]
            new_bboxes.append(new_bbox)

        self._log().debug("DEPAD BBOXES: {}".format(new_bboxes))

        return new_bboxes

    def _merge_tf_output(self, docling_output, pdf_cells):
        tf_output = []
        tf_cells_map = {}
        max_row_idx = 0

        for docling_item in docling_output:
            r_idx = str(docling_item["start_row_offset_idx"])
            c_idx = str(docling_item["start_col_offset_idx"])
            cell_key = c_idx + "_" + r_idx
            if cell_key in tf_cells_map:
                for pdf_cell in pdf_cells:
                    if pdf_cell["id"] == docling_item["cell_id"]:
                        text_cell_bbox = {
                            "b": pdf_cell["bbox"][3],
                            "l": pdf_cell["bbox"][0],
                            "r": pdf_cell["bbox"][2],
                            "t": pdf_cell["bbox"][1],
                            "token": pdf_cell["text"],
                        }
                        tf_cells_map[cell_key]["text_cell_bboxes"].append(
                            text_cell_bbox
                        )
            else:
                tf_cells_map[cell_key] = {
                    "bbox": docling_item["bbox"],
                    "row_span": docling_item["row_span"],
                    "col_span": docling_item["col_span"],
                    "start_row_offset_idx": docling_item["start_row_offset_idx"],
                    "end_row_offset_idx": docling_item["end_row_offset_idx"],
                    "start_col_offset_idx": docling_item["start_col_offset_idx"],
                    "end_col_offset_idx": docling_item["end_col_offset_idx"],
                    "indentation_level": docling_item["indentation_level"],
                    "text_cell_bboxes": [],
                    "column_header": docling_item["column_header"],
                    "row_header": docling_item["row_header"],
                    "row_section": docling_item["row_section"],
                }

                if docling_item["start_row_offset_idx"] > max_row_idx:
                    max_row_idx = docling_item["start_row_offset_idx"]

                for pdf_cell in pdf_cells:
                    if pdf_cell["id"] == docling_item["cell_id"]:
                        text_cell_bbox = {
                            "b": pdf_cell["bbox"][3],
                            "l": pdf_cell["bbox"][0],
                            "r": pdf_cell["bbox"][2],
                            "t": pdf_cell["bbox"][1],
                            "token": pdf_cell["text"],
                        }
                        tf_cells_map[cell_key]["text_cell_bboxes"].append(
                            text_cell_bbox
                        )

        for k in tf_cells_map:
            tf_output.append(tf_cells_map[k])
        return tf_output

    def resize_img(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]
        sf = 1.0
        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image, sf
        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            sf = r
            dim = (int(w * r), height)
        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            sf = r
            dim = (width, int(h * r))
        # resize the image
        # TODO(Nikos): Try to remove cv2 dependency
        resized = cv2.resize(image, dim, interpolation=inter)
        # return the resized image
        return resized, sf

    def multi_table_predict(
            self,
            page_inputs: list[dict],
            # one dict per page: {"image": np.ndarray, "tokens": [...], "width":..., "height":...}
            table_bboxes_list: list[list[list[float]]],
            # per-page list of [l,t,r,b] in ORIGINAL page coords (same as page_input["image"])
            do_matching: bool = True,
            correct_overlapping_cells: bool = False,
            sort_row_col_indexes: bool = True,
            doc_id: str = "unknown",
            start_page_idx: int = 0,
    ):
        """
        Batched API with ORIGINAL semantics:
          1) Resize each *page image* to height=1024 (record scale_factor).
          2) Scale each table bbox by the same page scale_factor.
          3) Crop table from the *resized* page.
          4) Run a single batched predict over all tables.
          5) Apply the original row/col reindexing or rs_seq counting.
        Returns a flat list of dicts [{ "tf_responses": ..., "predict_details": ... }, ...]
        in page order, table order (stable).
        """

        # -------- Phase 1: build per-table inputs with original resize semantics --------
        all_table_images: list[np.ndarray] = []
        all_scaled_bboxes: list[list[float]] = []
        all_scale_factors: list[float] = []
        all_iocr_pages: list[dict] = []

        for page_input, page_tbl_bboxes in zip(page_inputs, table_bboxes_list):
            # Original pipeline: resize the *page* once, then scale all table bboxes with the same factor
            page_image = page_input["image"]
            resized_page, scale_factor = self.resize_img(page_image, height=1024)

            for tbl_bbox in page_tbl_bboxes:
                # Scale bbox into resized-page coordinates (do not mutate caller data)
                x1 = tbl_bbox[0] * scale_factor
                y1 = tbl_bbox[1] * scale_factor
                x2 = tbl_bbox[2] * scale_factor
                y2 = tbl_bbox[3] * scale_factor
                scaled_bbox = [x1, y1, x2, y2]

                # Crop table region from the *resized* page (like standard)
                crop = resized_page[round(y1):round(y2), round(x1):round(x2)]

                all_table_images.append(crop)
                all_scaled_bboxes.append(scaled_bbox)
                all_scale_factors.append(scale_factor)
                all_iocr_pages.append(page_input)

        if not all_table_images:
            return []

        # -------- Phase 2: model + (optional) matching, batched --------
        # Uses your batched predict() that accepts lists. It will:
        #  - read scaled_bbox (resized space) and scale_factor
        #  - convert back to original page coords for matching
        batched_results = self.predict(
            iocr_pages=all_iocr_pages,
            table_bboxes=all_scaled_bboxes,
            table_images=all_table_images,
            scale_factors=all_scale_factors,
            eval_res_preds=None,
            correct_overlapping_cells=correct_overlapping_cells,
            do_matching=do_matching,
        )
        # batched_results: list of (tf_responses, matching_details)

        # -------- Phase 3: original post-step to finalize num_rows/num_cols & index compaction --------
        multi_tf_output: list[dict] = []

        for (tf_responses, predict_details) in batched_results:
            # predict_details here is the "matching_details" dict; we now augment it
            if sort_row_col_indexes:
                # Remap predicted start_row/col IDs to contiguous 0..K-1 indexes (original behavior)
                start_cols = []
                start_rows = []
                for c in tf_responses:
                    sc = c["start_col_offset_idx"]
                    sr = c["start_row_offset_idx"]
                    if sc not in start_cols:
                        start_cols.append(sc)
                    if sr not in start_rows:
                        start_rows.append(sr)
                start_cols.sort()
                start_rows.sort()

                max_end_c = 0
                max_end_r = 0
                for c in tf_responses:
                    c["start_col_offset_idx"] = start_cols.index(c["start_col_offset_idx"])
                    c["end_col_offset_idx"] = c["start_col_offset_idx"] + c["col_span"]
                    if c["end_col_offset_idx"] > max_end_c:
                        max_end_c = c["end_col_offset_idx"]

                    c["start_row_offset_idx"] = start_rows.index(c["start_row_offset_idx"])
                    c["end_row_offset_idx"] = c["start_row_offset_idx"] + c["row_span"]
                    if c["end_row_offset_idx"] > max_end_r:
                        max_end_r = c["end_row_offset_idx"]

                predict_details["num_cols"] = max_end_c
                predict_details["num_rows"] = max_end_r
            else:
                # Fallback identical to standard: infer from rs_seq when not compacting
                rs_seq = predict_details["prediction"]["rs_seq"] if "prediction" in predict_details else \
                    predict_details.get("prediction", {}).get("rs_seq", None)
                if rs_seq is None:
                    # If not present (e.g., legacy matching_details), keep conservative defaults
                    predict_details["num_cols"] = 0
                    predict_details["num_rows"] = 0
                else:
                    predict_details["num_cols"] = rs_seq.index("nl")
                    predict_details["num_rows"] = rs_seq.count("nl")

            multi_tf_output.append({
                "tf_responses": tf_responses,
                "predict_details": predict_details,
            })

        return multi_tf_output

    def predict(
            self,
            iocr_pages: list[dict],
            # per-table page_input dicts (must include "tokens" in original page coords if matching)
            table_bboxes: list[list[float]],
            # per-table bbox in RESIZED-PAGE coords (i.e., after page height=1024 scaling)
            table_images: list[np.ndarray],  # per-table crops from the RESIZED page (same coords as table_bboxes)
            scale_factors: list[float],  # per-table page resize factors (resized/original)
            eval_res_preds=None,
            correct_overlapping_cells: bool = False,
            do_matching: bool = True,
    ):
        """
        Faithful batched variant of the original single-table `predict`:
          - Preprocess table crops with the same Normalize+Resize pipeline
          - Call model once on a stacked batch
          - For each item: build prediction dict (bboxes/classes/tag_seq/rs_seq/html_seq), depad if needed
          - Check bbox/tag sync
          - Match cells (or dummy), optional post-process
          - Generate Docling responses, sort, and merge into tf_output
        Returns: list of (tf_output, matching_details) in the same order as inputs.
        """
        assert len(iocr_pages) == len(table_bboxes) == len(table_images) == len(scale_factors), \
            "All batched inputs must be same length"

        AggProfiler().start_agg(self._prof)

        # Create timer for detailed TF predictor timing
        # Fix: Check device type properly (handles 'cuda:0', 'cuda:1', etc.)
        is_cuda = str(self._device).startswith('cuda')
        timer = _CudaTimer() if is_cuda else _CPUTimer()

        # --- 1) Image preprocessing (original semantics, but batched) ---
        with timer.time_section('image_preprocessing'):
            resized_size = self._config["dataset"]["resized_image"]
            mean = self._config["dataset"]["image_normalization"]["mean"]
            std = self._config["dataset"]["image_normalization"]["std"]
            image_batch = self._batch_preprocess_images(table_images=table_images, resized_size=resized_size,
                                                        mean=mean, std=std)
            # # Equivalent to per-item: _prepare_image(table_image)
            # # i.e., Normalize -> Resize to (S,S) -> transpose to (C,W,H) -> /255 -> to(device) -> add batch dim
            # normalize = T.Normalize(
            #     mean=self._config["dataset"]["image_normalization"]["mean"],
            #     std=self._config["dataset"]["image_normalization"]["std"],
            # )
            # resized_size = self._config["dataset"]["resized_image"]
            # resize = T.Resize([resized_size, resized_size])
            #
            # batch_tensors = []
            # for table_img in table_images:
            #     img, _ = normalize(table_img, None)
            #     img, _ = resize(img, None)
            #     img = img.transpose(2, 1, 0)  # (channels, width, height)
            #     img = torch.FloatTensor(img / 255.0)  # float32 in [0,1]
            #     batch_tensors.append(img)
            #
            # if len(batch_tensors) == 0:
            #     return []
            #
            # image_batch = torch.stack(batch_tensors, dim=0)  # (N, C, W, H) — matches original quirk
            # image_batch = image_batch.to(device=self._device)

        # --- 2) Model inference (batched) ---
        with timer.time_section('model_inference'):
            max_steps = self._config["predict"]["max_steps"]
            beam_size = self._config["predict"]["beam_size"]

            all_predictions: list[dict] = []

            with torch.no_grad():
                if eval_res_preds is not None:
                    # Use provided eval predictions (expects a per-item list of dicts)
                    for ev in eval_res_preds:
                        pred = {
                            "bboxes": ev.get("bboxes", []),
                            "tag_seq": ev.get("tag_seq", []),
                        }
                        pred["rs_seq"] = self._get_html_tags(pred["tag_seq"])
                        pred["html_seq"] = otsl_to_html(pred["rs_seq"], False)
                        all_predictions.append(pred)
                else:
                    # Model may return batched tuples or per-item tuples; support both
                    model_result = self._model.predict(image_batch, max_steps, beam_size)

                # Normalize to per-item list of (seq, outputs_class, outputs_coord)
                def _normalize_model_batch_outputs(model_result):
                    # Already a per-item list of tuples
                    if isinstance(model_result, (list, tuple)) and model_result and isinstance(model_result[0],
                                                                                               (list, tuple)):
                        return list(model_result)
                    # Tuple of batched outputs
                    if isinstance(model_result, (list, tuple)) and len(model_result) in (2, 3):
                        if len(model_result) == 3:
                            seq_batch, class_batch, coord_batch = model_result
                        else:
                            seq_batch, class_batch, coord_batch = model_result[0], None, None
                        if not isinstance(seq_batch, (list, tuple)):
                            seq_batch = list(seq_batch)
                        out = []
                        for i in range(len(seq_batch)):
                            oc = class_batch[i] if class_batch is not None else None
                            od = coord_batch[i] if coord_batch is not None else None
                            out.append((seq_batch[i], oc, od))
                        return out
                    # Fallback: single item replicated?
                    return [(model_result, None, None)]

                triples = _normalize_model_batch_outputs(model_result)

                # Build original-format prediction dicts per item
                for (seq, outputs_class, outputs_coord) in triples:
                    pred = {}
                    # bboxes
                    if outputs_coord is not None:
                        if torch.is_tensor(outputs_coord) and outputs_coord.numel() == 0:
                            pred["bboxes"] = []
                        else:
                            bbox_xyxy = u.box_cxcywh_to_xyxy(outputs_coord)
                            pred["bboxes"] = bbox_xyxy.tolist() if torch.is_tensor(bbox_xyxy) else bbox_xyxy
                    else:
                        pred["bboxes"] = []
                    # classes
                    if outputs_class is not None:
                        if torch.is_tensor(outputs_class) and outputs_class.numel() > 0:
                            pred["classes"] = torch.argmax(outputs_class, dim=1).tolist()
                        elif isinstance(outputs_class, (list, tuple)):
                            pred["classes"] = list(outputs_class)
                        else:
                            pred["classes"] = []
                    else:
                        pred["classes"] = []
                    # tag seq (+ optional depadding)
                    if self._remove_padding:
                        seq, _ = u.remove_padding(seq)
                    pred["tag_seq"] = seq
                    pred["rs_seq"] = self._get_html_tags(seq)
                    pred["html_seq"] = otsl_to_html(pred["rs_seq"], False)

                    all_predictions.append(pred)

        # --- 3) Per-item: bbox/tag sync, matching, (optional) postproc, docling response, merge ---
        with timer.time_section('cell_matching_and_postprocess'):
            outputs: list[tuple[list[dict], dict]] = []

            for i, prediction in enumerate(all_predictions):
                iocr_page = iocr_pages[i]
                scaled_bbox = table_bboxes[i]  # coords in RESIZED-PAGE space
                scale_factor = scale_factors[i]

                # Logically identical checks to original
                self._log().debug("----- rs_seq -----")
                self._log().debug(prediction["rs_seq"])
                self._log().debug(len(prediction["rs_seq"]))
                otsl_sqr_chk(prediction["rs_seq"], False)

                # Sync bboxes vs tags
                with timer.time_section('bbox_sync'):
                    sync, corrected_bboxes = self._check_bbox_sync(prediction)
                    if not sync:
                        prediction["bboxes"] = corrected_bboxes

                # Prepare matching details
                matching_details = {
                    "table_cells": [],
                    "matches": {},
                    "pdf_cells": [],
                    "prediction_bboxes_page": [],
                }

                # Convert table bbox back to ORIGINAL page_input coords (undo page resize)
                tbl_bbox_for_match = [
                    scaled_bbox[0] / scale_factor,
                    scaled_bbox[1] / scale_factor,
                    scaled_bbox[2] / scale_factor,
                    scaled_bbox[3] / scale_factor,
                ]

                # Matching (faithful to original)
                if len(prediction["bboxes"]) > 0:
                    if do_matching:
                        with timer.time_section('match_cells'):
                            matching_details = self._cell_matcher.match_cells(
                                iocr_page, tbl_bbox_for_match, prediction
                            )
                        # Post-processing if tokens exist and post-process enabled
                        if len(iocr_page.get("tokens", [])) > 0 and self.enable_post_process:
                            with timer.time_section('post_process'):
                                AggProfiler().begin("post_process", self._prof)
                                matching_details = self._post_processor.process(
                                    matching_details, correct_overlapping_cells
                                )
                                AggProfiler().end("post_process", self._prof)
                    else:
                        with timer.time_section('match_cells_dummy'):
                            matching_details = self._cell_matcher.match_cells_dummy(
                                iocr_page, tbl_bbox_for_match, prediction
                            )

                # Generate Docling responses (as in original)
                with timer.time_section('generate_response'):
                    AggProfiler().begin("generate_docling_response", self._prof)
                    if do_matching:
                        docling_output = self._generate_tf_response(
                            matching_details["table_cells"], matching_details["matches"]
                        )
                    else:
                        docling_output = self._generate_tf_response_dummy(
                            matching_details["table_cells"]
                        )
                    AggProfiler().end("generate_docling_response", self._prof)

                # Sort and merge to TF output
                with timer.time_section('sort_and_merge'):
                    docling_output.sort(key=lambda item: item["cell_id"])
                    matching_details["docling_responses"] = docling_output
                    tf_output = self._merge_tf_output(docling_output, matching_details["pdf_cells"])

                outputs.append((tf_output, matching_details))
        
        # Finalize timing and log results
        timer.finalize()
        
        # Main breakdown
        print(f"     └─ tf predictor breakdown: preprocess={timer.get_time('image_preprocessing'):.1f}ms "
              f"inference={timer.get_time('model_inference'):.1f}ms "
              f"matching={timer.get_time('cell_matching_and_postprocess'):.1f}ms")
        
        # Detailed matching/postprocess breakdown
        bbox_sync_time = timer.get_time('bbox_sync')
        match_cells_time = timer.get_time('match_cells')
        match_dummy_time = timer.get_time('match_cells_dummy')
        post_process_time = timer.get_time('post_process')
        generate_response_time = timer.get_time('generate_response')
        sort_merge_time = timer.get_time('sort_and_merge')
        
        # Print detailed breakdown if any sub-timings exist
        if any([bbox_sync_time, match_cells_time, match_dummy_time, post_process_time, generate_response_time, sort_merge_time]):
            print(f"        ├─ bbox_sync: {bbox_sync_time:.1f}ms")
            if match_cells_time > 0:
                print(f"        ├─ match_cells: {match_cells_time:.1f}ms")
            if match_dummy_time > 0:
                print(f"        ├─ match_cells_dummy: {match_dummy_time:.1f}ms")
            if post_process_time > 0:
                print(f"        ├─ post_process: {post_process_time:.1f}ms")
            print(f"        ├─ generate_response: {generate_response_time:.1f}ms")
            print(f"        └─ sort_and_merge: {sort_merge_time:.1f}ms")

        return outputs

    def _generate_tf_response_dummy(self, table_cells):
        tf_cell_list = []

        for table_cell in table_cells:
            colspan_val = 1
            if "colspan_val" in table_cell:
                colspan_val = table_cell["colspan_val"]
            rowspan_val = 1
            if "rowspan_val" in table_cell:
                rowspan_val = table_cell["rowspan_val"]

            column_header = False
            if table_cell["label"] == "ched":
                column_header = True

            row_header = False
            if table_cell["label"] == "rhed":
                row_header = True

            row_section = False
            if table_cell["label"] == "srow":
                row_section = True

            row_id = table_cell["row_id"]
            column_id = table_cell["column_id"]

            cell_bbox = {
                "b": table_cell["bbox"][3],
                "l": table_cell["bbox"][0],
                "r": table_cell["bbox"][2],
                "t": table_cell["bbox"][1],
                "token": "",
            }

            tf_cell = {
                "cell_id": table_cell["cell_id"],
                "bbox": cell_bbox,  # b,l,r,t,token
                "row_span": rowspan_val,
                "col_span": colspan_val,
                "start_row_offset_idx": row_id,
                "end_row_offset_idx": row_id + rowspan_val,
                "start_col_offset_idx": column_id,
                "end_col_offset_idx": column_id + colspan_val,
                "indentation_level": 0,
                # No text cell bboxes, because no matching was done
                "text_cell_bboxes": [],
                "column_header": column_header,
                "row_header": row_header,
                "row_section": row_section,
            }
            tf_cell_list.append(tf_cell)
        return tf_cell_list

    def _generate_tf_response(self, table_cells, matches):
        r"""
        Convert the matching details to the expected output for Docling

        Parameters
        ----------
        table_cells : list of dict
            Each value is a dictionary with keys: "cell_id", "row_id", "column_id",
                                                  "bbox", "label", "class"
        matches : dictionary of lists of table_cells
            A dictionary which is indexed by the pdf_cell_id as key and the value is a list
            of the table_cells that fall inside that pdf cell

        Returns
        -------
        docling_output : string
            json response formatted according to Docling api expectations
        """

        # format output to look similar to tests/examples/tf_gte_output_2.json
        tf_cell_list = []
        for pdf_cell_id, pdf_cell_matches in matches.items():
            tf_cell = {
                "bbox": {},  # b,l,r,t,token
                "row_span": 1,
                "col_span": 1,
                "start_row_offset_idx": -1,
                "end_row_offset_idx": -1,
                "start_col_offset_idx": -1,
                "end_col_offset_idx": -1,
                "indentation_level": 0,
                # return text cell bboxes additionally to the matched index
                "text_cell_bboxes": [{}],  # b,l,r,t,token
                "column_header": False,
                "row_header": False,
                "row_section": False,
            }
            tf_cell["cell_id"] = int(pdf_cell_id)

            row_ids = set()
            column_ids = set()
            labels = set()

            for match in pdf_cell_matches:
                tm = match["table_cell_id"]
                tcl = list(
                    filter(lambda table_cell: table_cell["cell_id"] == tm, table_cells)
                )
                if len(tcl) > 0:
                    table_cell = tcl[0]
                    row_ids.add(table_cell["row_id"])
                    column_ids.add(table_cell["column_id"])
                    labels.add(table_cell["label"])

                    if table_cell["label"] is not None:
                        if table_cell["label"] in ["ched"]:
                            tf_cell["column_header"] = True
                        if table_cell["label"] in ["rhed"]:
                            tf_cell["row_header"] = True
                        if table_cell["label"] in ["srow"]:
                            tf_cell["row_section"] = True

                    tf_cell["start_col_offset_idx"] = table_cell["column_id"]
                    tf_cell["end_col_offset_idx"] = table_cell["column_id"] + 1
                    tf_cell["start_row_offset_idx"] = table_cell["row_id"]
                    tf_cell["end_row_offset_idx"] = table_cell["row_id"] + 1

                    if "colspan_val" in table_cell:
                        tf_cell["col_span"] = table_cell["colspan_val"]
                        tf_cell["start_col_offset_idx"] = table_cell["column_id"]
                        off_idx = table_cell["column_id"] + tf_cell["col_span"]
                        tf_cell["end_col_offset_idx"] = off_idx
                    if "rowspan_val" in table_cell:
                        tf_cell["row_span"] = table_cell["rowspan_val"]
                        tf_cell["start_row_offset_idx"] = table_cell["row_id"]
                        tf_cell["end_row_offset_idx"] = (
                            table_cell["row_id"] + tf_cell["row_span"]
                        )
                    if "bbox" in table_cell:
                        table_match_bbox = table_cell["bbox"]
                        tf_bbox = {
                            "b": table_match_bbox[3],
                            "l": table_match_bbox[0],
                            "r": table_match_bbox[2],
                            "t": table_match_bbox[1],
                        }
                        tf_cell["bbox"] = tf_bbox

            tf_cell["row_ids"] = list(row_ids)
            tf_cell["column_ids"] = list(column_ids)
            tf_cell["label"] = "None"
            l_labels = list(labels)
            if len(l_labels) > 0:
                tf_cell["label"] = l_labels[0]
            tf_cell_list.append(tf_cell)
        return tf_cell_list

    def _prepare_image(self, mat_image):
        r"""
        Rescale the image and prepare a batch of 1 with the image as as tensor

        Parameters
        ----------
        mat_image: cv2.Mat
            The image as an openCV Mat object

        Returns
        -------
        tensor (batch_size, image_channels, resized_image, resized_image)
        """
        normalize = T.Normalize(
            mean=self._config["dataset"]["image_normalization"]["mean"],
            std=self._config["dataset"]["image_normalization"]["std"],
        )
        resized_size = self._config["dataset"]["resized_image"]
        resize = T.Resize([resized_size, resized_size])

        img, _ = normalize(mat_image, None)
        img, _ = resize(img, None)

        img = img.transpose(2, 1, 0)  # (channels, width, height)
        img = torch.FloatTensor(img / 255.0)
        image_batch = img.unsqueeze(dim=0)
        image_batch = image_batch.to(device=self._device)
        return image_batch

    def _get_html_tags(self, seq):
        r"""
        Convert indices to actual html tags

        """
        # Map the tag indices back to actual tags (without start, end)
        html_tags = [self._rev_word_map[ind] for ind in seq[1:-1]]

        return html_tags

    def _batch_preprocess_images(self, table_images: List[np.ndarray], resized_size: int,  mean, std, dtype=torch.float32, use_stream=True):
        """Batch preprocess images"""
        if not table_images:
            return []

            # --- bucket by (H, W, C) so each bucket stacks cleanly ---
        buckets = {}  # (H, W, C) -> [indices]
        shapes = []
        for i, img in enumerate(table_images):
            if img.ndim == 2:
                # promote grayscale to HxWx1 to avoid surprises
                img = img[..., None]
                table_images[i] = img
            if img.dtype != np.uint8:
                table_images[i] = img.astype(np.uint8, copy=False)
            h, w, c = img.shape
            key = (h, w, c)
            buckets.setdefault(key, []).append(i)
            shapes.append(key)

        # sanity: channel count should be consistent for your pipeline
        first_c = shapes[0][2]
        if any(c != first_c for (_, _, c) in shapes):
            raise ValueError(f"Inconsistent channel counts detected: {[s[2] for s in shapes]}")

        N, C, S = len(table_images), first_c, int(resized_size)
        out = torch.empty((N, C, S, S), device=self._device, dtype=dtype)

        # prepare normalization buffers once
        mean_t = torch.tensor(mean, device=self._device, dtype=dtype).view(1, 1, 1, C)
        std_t = torch.tensor(std, device=self._device, dtype=dtype).view(1, 1, 1, C)

        stream = torch.cuda.Stream(device=self._device) if use_stream else torch.cuda.current_stream(device=self._device)
        with torch.cuda.stream(stream):
            for (h, w, c), idxs in buckets.items():
                # stack this bucket on CPU (identical shapes)
                nhwc = np.stack([table_images[i] for i in idxs], axis=0)  # [B,H,W,C], uint8
                cpu = torch.from_numpy(nhwc).pin_memory()

                # H2D once, normalize, resize on GPU
                t = cpu.to(device=self._device, dtype=dtype, non_blocking=True)  # [B,H,W,C]
                t = t / 255.0
                t = (t - mean_t) / std_t

                # NCHW for interpolate
                t = t.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)  # [B,C,H,W]
                t = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)

                # your layout quirk: (C, W, H). Swap the last two dims.
                t = t.permute(0, 1, 3, 2).contiguous()  # [B,C,W,H]

                # place back in original order
                out[torch.as_tensor(idxs, device=self._device, dtype=torch.long)] = t

        if use_stream:
            torch.cuda.current_stream(device=self._device).wait_stream(stream)

        return out  # (N, C, W, H), float32 on device

