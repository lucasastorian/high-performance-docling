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
from pathlib import Path
import hashlib
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_model

import docling_ibm_models.tableformer.common as c
import docling_ibm_models.tableformer.settings as s
import docling_ibm_models.tableformer.utils.utils as u
from docling_ibm_models.tableformer.otsl import otsl_to_html
from docling_ibm_models.tableformer.utils.app_profiler import AggProfiler

from tf_cell_matcher import CellMatcher
from tablemodel04_rs import TableModel04_rs
from table_timing_debug import get_timing_collector
from matching_post_processor import MatchingPostProcessor
import transforms as T

# LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG
LOG_LEVEL = logging.WARN

logger = s.get_custom_logger(__name__, LOG_LEVEL)

# Global lock for model initialization to prevent threading issues
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

        # Caching configuration for baseline collection
        self.enable_cache = False  # Set to True to enable caching
        self.cache_dir = None  # Will be set when cache is enabled
        self.cache_metadata = []  # Stores metadata for cached items

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

            if model is None:
                err_msg = "Not able to initiate a model for {}".format(self._model_type)
                self._log().error(err_msg)
                raise ValueError(err_msg)

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

        return model

    def get_device(self):
        return self._device

    def get_model_type(self):
        return self._model_type

    def _log(self):
        # Setup a custom logger
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    def enable_baseline_cache(self, cache_dir="./tf_cache", mode="baseline"):
        """
        Enable caching for data collection.

        Args:
            cache_dir: Root cache directory
            mode: "baseline" (overwrites) or "run" (creates timestamped folder)
        """
        self.enable_cache = True
        self.cache_mode = mode

        if mode == "baseline":
            # Baseline mode - always use same directory, overwrite
            self.cache_dir = Path(cache_dir) / "baseline"
            # Clear existing baseline
            if self.cache_dir.exists():
                import shutil
                shutil.rmtree(self.cache_dir)
                self._log().info(f"Cleared existing baseline at: {self.cache_dir}")
        else:
            # Run mode - create timestamped directory
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.cache_dir = Path(cache_dir) / "runs" / timestamp

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.cache_dir / "crops").mkdir(exist_ok=True)
        (self.cache_dir / "tensors").mkdir(exist_ok=True)
        (self.cache_dir / "outputs").mkdir(exist_ok=True)

        self._log().info(f"Cache enabled ({mode} mode) at: {self.cache_dir}")

    @staticmethod
    def get_latest_run(cache_dir="./tf_cache"):
        """Get the path to the latest run directory."""
        runs_dir = Path(cache_dir) / "runs"
        if not runs_dir.exists():
            return None

        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            return None

        # Sort by name (timestamp) and return latest
        return sorted(run_dirs)[-1]

    @staticmethod
    def get_baseline_dir(cache_dir="./tf_cache"):
        """Get the path to the baseline directory."""
        return Path(cache_dir) / "baseline"

    def _save_cache_entry(self, doc_id, page_idx, table_idx, table_image,
                          prepared_tensor, outputs, table_bbox, scaled_bbox, scale_factor):
        """Save a cache entry for baseline comparison."""
        if not self.enable_cache:
            return

        # Generate hash for the image
        img_bytes = cv2.imencode('.png', table_image)[1].tobytes()
        img_hash = hashlib.sha256(img_bytes).hexdigest()[:8]

        # File names
        base_name = f"{doc_id}_p{page_idx}_t{table_idx}_{img_hash}"
        crop_path = self.cache_dir / "crops" / f"{base_name}.png"
        tensor_path = self.cache_dir / "tensors" / f"{base_name}.pt"
        output_path = self.cache_dir / "outputs" / f"{base_name}.json"

        # Save crop
        cv2.imwrite(str(crop_path), table_image)

        # Save prepared tensor
        torch.save(prepared_tensor.cpu(), tensor_path)

        # Save outputs
        output_data = {
            "tag_seq": outputs.get("tag_seq", []),
            "html_seq": outputs.get("html_seq", []),
            "rs_seq": outputs.get("rs_seq", []),
            "num_bboxes": len(outputs.get("bboxes", [])),
            "timestamp": datetime.now().isoformat()
        }

        # Convert tensors to lists for JSON serialization
        if "outputs_class" in outputs and outputs["outputs_class"] is not None:
            if torch.is_tensor(outputs["outputs_class"]):
                output_data["outputs_class"] = outputs["outputs_class"].cpu().tolist()
            else:
                output_data["outputs_class"] = outputs["outputs_class"]  # Already a list

        if "outputs_coord" in outputs and outputs["outputs_coord"] is not None:
            if torch.is_tensor(outputs["outputs_coord"]):
                output_data["outputs_coord"] = outputs["outputs_coord"].cpu().tolist()
            else:
                output_data["outputs_coord"] = outputs["outputs_coord"]  # Already a list

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        # Add metadata entry
        metadata = {
            "doc_id": doc_id,
            "page_index": page_idx,
            "table_index": table_idx,
            "orig_bbox": table_bbox,
            "scaled_bbox": scaled_bbox,
            "scale_factor": scale_factor,
            "image_path": str(crop_path.relative_to(self.cache_dir)),
            "tensor_path": str(tensor_path.relative_to(self.cache_dir)),
            "output_path": str(output_path.relative_to(self.cache_dir)),
            "prepared_size": list(prepared_tensor.shape),
            "sha256": img_hash,
            "timestamp": datetime.now().isoformat()
        }
        self.cache_metadata.append(metadata)

        # Save metadata file
        metadata_path = self.cache_dir / "metadata.jsonl"
        with open(metadata_path, 'a') as f:
            f.write(json.dumps(metadata) + '\n')

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

    def _cache_prediction(self, predict_details, doc_id, page_idx, table_idx, table_image, prepared_tensor,
                          orig_table_bbox, table_bbox, scale_factor):
        """Caches the prediction"""
        cache_outputs = {
            "tag_seq": predict_details.get("tag_seq", []),
            "rs_seq": predict_details.get("rs_seq", []),
            "html_seq": predict_details.get("html_seq", []),
            "bboxes": predict_details.get("bboxes", []),
            "outputs_class": predict_details.get("classes", []),
            "outputs_coord": predict_details.get("bboxes", [])
        }

        self._save_cache_entry(
            doc_id=doc_id,
            page_idx=page_idx,
            table_idx=table_idx,
            table_image=table_image,
            prepared_tensor=prepared_tensor,
            outputs=cache_outputs,
            table_bbox=orig_table_bbox,
            scaled_bbox=table_bbox,
            scale_factor=scale_factor
        )

    def predict_dummy(
            self, iocr_page, table_bbox, table_image, scale_factor, eval_res_preds=None
    ):
        r"""
        Predict the table out of an image in memory

        Parameters
        ----------
        iocr_page : dict
            Docling provided table data
        eval_res_preds : dict
            Ready predictions provided by the evaluation results

        Returns
        -------
        docling_output : string
            json response formatted according to Docling api expectations

        matching_details : string
            json with details about the matching between the pdf cells and the table cells
        """
        AggProfiler().start_agg(self._prof)

        max_steps = self._config["predict"]["max_steps"]
        beam_size = self._config["predict"]["beam_size"]
        image_batch = self._prepare_image(table_image)
        # Make predictions
        prediction = {}

        with torch.no_grad():
            # Compute predictions
            if (
                    eval_res_preds is not None
            ):  # Don't run the model, use the provided predictions
                prediction["bboxes"] = eval_res_preds["bboxes"]
                pred_tag_seq = eval_res_preds["tag_seq"]
            elif self._config["predict"]["bbox"]:
                start = datetime.now()
                pred_tag_seq, outputs_class, outputs_coord = self._model.predict(
                    image_batch, max_steps, beam_size
                )
                print(f"Executed table predictions in {datetime.now() - start} seconds")

                if outputs_coord is not None:
                    if len(outputs_coord) == 0:
                        prediction["bboxes"] = []
                    else:
                        bbox_pred = u.box_cxcywh_to_xyxy(outputs_coord)
                        prediction["bboxes"] = bbox_pred.tolist()
                else:
                    prediction["bboxes"] = []

                if outputs_class is not None:
                    if len(outputs_class) == 0:
                        prediction["classes"] = []
                    else:
                        result_class = torch.argmax(outputs_class, dim=1)
                        prediction["classes"] = result_class.tolist()
                else:
                    prediction["classes"] = []
                if self._remove_padding:
                    pred_tag_seq, _ = u.remove_padding(pred_tag_seq)
            else:
                pred_tag_seq, _, _ = self._model.predict(
                    image_batch, max_steps, beam_size
                )
                # Check if padding should be removed
                if self._remove_padding:
                    pred_tag_seq, _ = u.remove_padding(pred_tag_seq)

            prediction["tag_seq"] = pred_tag_seq
            prediction["rs_seq"] = self._get_html_tags(pred_tag_seq)
            prediction["html_seq"] = otsl_to_html(prediction["rs_seq"], False)
        # Remove implied padding from bbox predictions,
        # that we added on image pre-processing stage
        self._log().debug("----- rs_seq -----")
        self._log().debug(prediction["rs_seq"])
        self._log().debug(len(prediction["rs_seq"]))
        otsl_sqr_chk(prediction["rs_seq"], False)

        # Check that bboxes are in sync with predicted tags
        sync, corrected_bboxes = self._check_bbox_sync(prediction)
        if not sync:
            prediction["bboxes"] = corrected_bboxes

        # Match the cells
        matching_details = {
            "table_cells": [],
            "matches": {},
            "pdf_cells": [],
            "prediction_bboxes_page": [],
        }

        # Table bbox upscaling will scale predicted bboxes too within cell matcher
        scaled_table_bbox = [
            table_bbox[0] / scale_factor,
            table_bbox[1] / scale_factor,
            table_bbox[2] / scale_factor,
            table_bbox[3] / scale_factor,
        ]

        if len(prediction["bboxes"]) > 0:
            matching_details = self._cell_matcher.match_cells_dummy(
                iocr_page, scaled_table_bbox, prediction
            )
            # Generate the expected Docling responses
            AggProfiler().begin("generate_docling_response", self._prof)
            docling_output = self._generate_tf_response_dummy(
                matching_details["table_cells"]
            )

            AggProfiler().end("generate_docling_response", self._prof)
            # Add the docling_output sorted by cell_id into the matching_details
            docling_output.sort(key=lambda item: item["cell_id"])
            matching_details["docling_responses"] = docling_output
            # Merge docling_output and pdf_cells into one TF output,
            # with deduplicated table cells
            # tf_output = self._merge_tf_output_dummy(docling_output)
            tf_output = docling_output

        return tf_output, matching_details

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
            tf_cell = {"bbox": {}, "row_span": 1, "col_span": 1, "start_row_offset_idx": -1, "end_row_offset_idx": -1,
                       "start_col_offset_idx": -1, "end_col_offset_idx": -1, "indentation_level": 0,
                       "text_cell_bboxes": [{}], "column_header": False, "row_header": False, "row_section": False,
                       "cell_id": int(pdf_cell_id)}

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

    # -----------------------
    # Helpers (private)
    # -----------------------

    def _preprocess_table(self, page_image: np.ndarray, table_bbox: list[float]):
        """
        Given the original page image and an unscaled table_bbox expressed in the
        page_input coordinate system, resize the page to height=1024, scale the bbox
        to the resized image, and return (table_crop, scaled_bbox, scale_factor).
        """
        page_image_resized, scale_factor = self.resize_img(page_image, height=1024)

        # Scale bbox to resized-image coordinates (do not mutate caller input)
        sx1 = table_bbox[0] * scale_factor
        sy1 = table_bbox[1] * scale_factor
        sx2 = table_bbox[2] * scale_factor
        sy2 = table_bbox[3] * scale_factor
        scaled_bbox = [sx1, sy1, sx2, sy2]

        # Crop the table region from resized image
        table_crop = page_image_resized[
                     round(sy1): round(sy2),
                     round(sx1): round(sx2),
                     ]

        return table_crop, scaled_bbox, scale_factor

    # def _prepare_image_batch(self, images: list[np.ndarray]) -> torch.Tensor:
    #     """
    #     Convert a list of OpenCV images to a single batch tensor on the right device.
    #     """
    #     if not images:
    #         return torch.empty(
    #             0, 3, self._config["dataset"]["resized_image"], self._config["dataset"]["resized_image"]
    #         ).to(self._device)
    #
    #     normalize = T.Normalize(
    #         mean=self._config["dataset"]["image_normalization"]["mean"],
    #         std=self._config["dataset"]["image_normalization"]["std"],
    #     )
    #     resized_size = self._config["dataset"]["resized_image"]
    #     resize = T.Resize([resized_size, resized_size])
    #
    #     batch = []
    #     for img in images:
    #         img, _ = normalize(img, None)
    #         img, _ = resize(img, None)
    #         img = img.transpose(2, 1, 0)  # (C, W, H)
    #         img = torch.FloatTensor(img / 255.0)
    #         batch.append(img)
    #
    #     batch_tensor = torch.stack(batch, dim=0).to(self._device)
    #
    #     return batch_tensor

    def _prepare_image_batch(self, images: list[np.ndarray]) -> torch.Tensor:
        """
        Full-GPU batch prep with semantics equivalent to:
          x = F.normalize(img, mean, std)  # (img - 255*mean)/std  in 0..255
          x = cv2.resize(x, (S,S), INTER_LINEAR)
          x = x.transpose(2,1,0)           # (C, W, H)
          x = x / 255.0
        which is identical to: ((img/255) - mean) / std, then resize, then (C,W,H).
        """
        S = self._config["dataset"]["resized_image"]
        if not images:
            return torch.empty(0, 3, S, S, device=self._device, dtype=torch.float32)

        # mean/std are in 0..1 domain (ImageNet-style)
        mean = torch.tensor(self._config["dataset"]["image_normalization"]["mean"],
                            device=self._device, dtype=torch.float32).view(1, 1, 3)  # HWC broadcast
        std = torch.tensor(self._config["dataset"]["image_normalization"]["std"],
                           device=self._device, dtype=torch.float32).view(1, 1, 3)

        out = torch.empty(len(images), 3, S, S, device=self._device, dtype=torch.float32)

        for i, np_img in enumerate(images):
            # Defensive channel handling: grayscale -> 3ch, drop alpha
            if np_img.ndim == 2:
                np_img = np.repeat(np_img[..., None], 3, axis=-1)
            elif np_img.shape[-1] > 3:
                np_img = np_img[..., :3]

            # HWC uint8/float -> HWC float32 on GPU, scaled to [0,1]
            t = torch.from_numpy(np_img).to(self._device)
            if t.dtype != torch.float32:
                t = t.to(torch.float32)
            t = t / 255.0

            # Normalize in 0..1 domain (equivalent to your (img - 255*mean)/std then /255 later)
            t = (t - mean) / std  # HWC

            # Resize to S×S (bilinear, align_corners=False ~ OpenCV INTER_LINEAR)
            t = t.permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
            t = F.interpolate(t, size=(S, S), mode="bilinear", align_corners=False)
            t = t.squeeze(0)  # C,H,S

            # Final layout quirk: (C, W, H) — keep it because the model was trained this way
            t = t.permute(0, 2, 1)  # C,W,H

            out[i].copy_(t)

        return out.contiguous(memory_format=torch.channels_last)

    def _normalize_model_batch_outputs(self, model_result):
        """
        Normalize various possible return shapes from TableModel04_rs.predict into
        a list of per-item tuples: [(seq, outputs_class_i, outputs_coord_i), ...]
        """
        # Case A: iterable of per-item tuples already
        if isinstance(model_result, (list, tuple)) and model_result and isinstance(model_result[0], (list, tuple)):
            return list(model_result)

        # Case B: tuple of batched things (seq_batch, class_batch, coord_batch)
        if isinstance(model_result, (list, tuple)) and len(model_result) in (2, 3):
            if len(model_result) == 3:
                seq_batch, class_batch, coord_batch = model_result
            else:
                # bbox disabled path often returns (seq_batch, _, _)
                seq_batch, class_batch, coord_batch = model_result[0], None, None

            # Ensure list-like
            if not isinstance(seq_batch, (list, tuple)):
                seq_batch = list(seq_batch)

            triples = []
            for i in range(len(seq_batch)):
                oc = class_batch[i] if class_batch is not None else None
                od = coord_batch[i] if coord_batch is not None else None
                triples.append((seq_batch[i], oc, od))
            return triples

        # Fallback: assume it's a single sequence
        return [(model_result, None, None)]

    def _build_prediction_from_heads(self, seq, outputs_class, outputs_coord):
        """
        Turn raw model-head outputs into the 'prediction' dict expected downstream.
        """
        pred = {}

        # bboxes
        if outputs_coord is not None:
            if torch.is_tensor(outputs_coord) and outputs_coord.numel() == 0:
                pred["bboxes"] = []
            else:
                # outputs_coord can be a tensor or list; convert per-item to xyxy float lists
                bbox_xyxy = u.box_cxcywh_to_xyxy(outputs_coord)
                if torch.is_tensor(bbox_xyxy):
                    pred["bboxes"] = bbox_xyxy.tolist()
                else:
                    pred["bboxes"] = bbox_xyxy
        else:
            pred["bboxes"] = []

        # classes
        if outputs_class is not None:
            if torch.is_tensor(outputs_class) and outputs_class.numel() > 0:
                pred["classes"] = torch.argmax(outputs_class, dim=1).tolist()
            elif isinstance(outputs_class, (list, tuple)):
                # already per-item
                pred["classes"] = list(outputs_class)
            else:
                pred["classes"] = []
        else:
            pred["classes"] = []

        # tag seq
        if self._remove_padding:
            seq, _ = u.remove_padding(seq)

        pred["tag_seq"] = seq
        pred["rs_seq"] = self._get_html_tags(seq)
        pred["html_seq"] = otsl_to_html(pred["rs_seq"], False)

        return pred

    def _finalize_predict_details(self, tf_responses: list[dict], predict_details: dict, sort_row_col_indexes: bool):
        """
        Compute num_rows/num_cols either by remapping IDs to contiguous indexes
        or by using the rs_seq fallback.
        """
        if sort_row_col_indexes:
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
                max_end_c = max(max_end_c, c["end_col_offset_idx"])

                c["start_row_offset_idx"] = start_rows.index(c["start_row_offset_idx"])
                c["end_row_offset_idx"] = c["start_row_offset_idx"] + c["row_span"]
                max_end_r = max(max_end_r, c["end_row_offset_idx"])

            predict_details["num_cols"] = max_end_c
            predict_details["num_rows"] = max_end_r
        else:
            rs_seq = predict_details["prediction"]["rs_seq"]
            predict_details["num_cols"] = rs_seq.index("nl")
            predict_details["num_rows"] = rs_seq.count("nl")

        return predict_details

    # -----------------------
    # Batched predict (public)
    # -----------------------

    def predict(
            self,
            iocr_pages: list[dict],
            table_bboxes: list[list[float]],  # bboxes scaled to resized image coords
            table_images: list[np.ndarray],  # cropped table images from resized page
            scale_factors: list[float],
            eval_res_preds=None,
            correct_overlapping_cells: bool = False,
            do_matching: bool = True,
    ):
        """
        Batched predict over N tables. Returns a list of (tf_output, matching_details).
        - iocr_pages: per-table page_input dicts (must include tokens in original page coords)
        - table_bboxes: per-table bbox in RESIZED image coords (same space as table_images)
        - scale_factors: per-table resize scale factor (resized / original)
        """
        AggProfiler().start_agg(self._prof)

        timer = get_timing_collector()

        max_steps = self._config["predict"]["max_steps"]
        beam_size = self._config["predict"]["beam_size"]

        timer.start("prepare_image_batch")
        image_batch = self._prepare_image_batch(table_images)
        timer.end("prepare_image_batch")

        all_predictions: list[dict] = []

        with torch.no_grad():
            if eval_res_preds is not None:
                for ev in eval_res_preds:
                    pred = {
                        "bboxes": ev.get("bboxes", []),
                        "tag_seq": ev.get("tag_seq", []),
                    }
                    pred["rs_seq"] = self._get_html_tags(pred["tag_seq"])
                    pred["html_seq"] = otsl_to_html(pred["rs_seq"], False)
                    all_predictions.append(pred)
            else:
                start = datetime.now()
                timer.start("model_inference")
                model_result = self._model.predict(image_batch, max_steps, beam_size)
                timer.end("model_inference")
                print(f"Made table predictions in {datetime.now() - start} seconds")

                timer.start("normalize_outputs")
                triples = self._normalize_model_batch_outputs(model_result)
                timer.end("normalize_outputs")
                for seq, outputs_class, outputs_coord in triples:
                    pred = self._build_prediction_from_heads(seq, outputs_class, outputs_coord)
                    all_predictions.append(pred)

        outputs = []
        for i, prediction in enumerate(all_predictions):
            iocr_page = iocr_pages[i]
            scaled_bbox = table_bboxes[i]  # coords in resized-image space
            scale_factor = scale_factors[i]

            # Basic checks
            self._log().debug("----- rs_seq -----")
            self._log().debug(prediction["rs_seq"])
            self._log().debug(len(prediction["rs_seq"]))
            otsl_sqr_chk(prediction["rs_seq"], False)

            # Sync bboxes vs tags
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

            # Convert table bbox back to original page_input coordinate space
            tbl_bbox_for_match = [
                scaled_bbox[0] / scale_factor,
                scaled_bbox[1] / scale_factor,
                scaled_bbox[2] / scale_factor,
                scaled_bbox[3] / scale_factor,
            ]

            # Matching
            if len(prediction["bboxes"]) > 0:
                if do_matching:
                    timer.start("match_cells")
                    matching_details = self._cell_matcher.match_cells(
                        iocr_page, tbl_bbox_for_match, prediction
                    )
                    timer.end("match_cells")
                    if len(iocr_page.get("tokens", [])) > 0 and self.enable_post_process:
                        AggProfiler().begin("post_process", self._prof)
                        timer.start("post_process")
                        matching_details = self._post_processor.process(
                            matching_details, correct_overlapping_cells
                        )
                        timer.end("post_process")
                        AggProfiler().end("post_process", self._prof)
                else:
                    timer.start("match_cells_dummy")
                    matching_details = self._cell_matcher.match_cells_dummy(
                        iocr_page, tbl_bbox_for_match, prediction
                    )
                    timer.end("match_cells_dummy")

            # Attach prediction for downstream consumers
            matching_details["prediction"] = prediction

            # Generate Docling responses
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

            # Sort and merge
            docling_output.sort(key=lambda item: item["cell_id"])
            matching_details["docling_responses"] = docling_output
            tf_output = self._merge_tf_output(docling_output, matching_details["pdf_cells"])

            outputs.append((tf_output, matching_details))

        # Don't print here - let multi_table_predict handle it

        return outputs

    # -----------------------
    # Top-level entry (public)
    # -----------------------

    def multi_table_predict(
            self,
            page_inputs: list[dict],  # one dict per page
            table_bboxes_list: list[list[list[float]]],  # list of bbox lists, per page
            do_matching: bool = True,
            correct_overlapping_cells: bool = False,
            sort_row_col_indexes: bool = True,
            doc_id: str = "unknown",
            start_page_idx: int = 0,
    ):
        """
        Batch over pages, then batch over all tables. Stable mapping and no coordinate shenanigans.
        Expects page_inputs[i]["tokens"] populated in ORIGINAL page_input coords.
        """
        # Reset timing collector for this run
        from table_timing_debug import reset_timing
        reset_timing()
        timer = get_timing_collector()

        # Phase 1: collect all tables
        timer.start("phase1_collect_tables")
        all_table_images: list[np.ndarray] = []
        all_scaled_bboxes: list[list[float]] = []
        all_scale_factors: list[float] = []
        all_iocr_pages: list[dict] = []
        meta: list[dict] = []

        for rel_page_idx, (page_input, page_tbl_bboxes) in enumerate(zip(page_inputs, table_bboxes_list)):
            page_image = page_input["image"]
            # Pre-resize once per page
            timer.start("resize_page_image")
            page_image_resized, scale_factor = self.resize_img(page_image, height=1024)
            timer.end("resize_page_image")

            timer.start("crop_tables")
            for table_idx, tbl_bbox in enumerate(page_tbl_bboxes):
                # Work from a copy; keep caller data pristine
                scaled_crop_bbox = [
                    tbl_bbox[0] * scale_factor,
                    tbl_bbox[1] * scale_factor,
                    tbl_bbox[2] * scale_factor,
                    tbl_bbox[3] * scale_factor,
                ]
                table_crop = page_image_resized[
                             round(scaled_crop_bbox[1]): round(scaled_crop_bbox[3]),
                             round(scaled_crop_bbox[0]): round(scaled_crop_bbox[2]),
                             ]

                all_table_images.append(table_crop)
                all_scaled_bboxes.append(scaled_crop_bbox)
                all_scale_factors.append(scale_factor)
                all_iocr_pages.append(page_input)
                meta.append({
                    "page_idx": start_page_idx + rel_page_idx,
                    "table_idx": table_idx,
                    "doc_id": doc_id,
                })
            timer.end("crop_tables")
        timer.end("phase1_collect_tables")

        if not all_table_images:
            return []

        # Phase 2: model + matching
        timer.start("phase2_predict")
        batched_results = self.predict(
            iocr_pages=all_iocr_pages,
            table_bboxes=all_scaled_bboxes,
            table_images=all_table_images,
            scale_factors=all_scale_factors,
            correct_overlapping_cells=correct_overlapping_cells,
            do_matching=do_matching,
        )
        timer.end("phase2_predict")

        # Phase 3: package outputs per table (order preserved)
        timer.start("phase3_package_outputs")
        multi_tf_output: list[dict] = []
        for i, (tf_responses, matching_details) in enumerate(batched_results):
            # Compute row/col counts and compact indexes if requested
            predict_details = matching_details  # naming compatibility
            timer.start("finalize_predict_details")
            predict_details = self._finalize_predict_details(
                tf_responses=tf_responses,
                predict_details=predict_details,
                sort_row_col_indexes=sort_row_col_indexes,
            )
            timer.end("finalize_predict_details")

            # Cache if needed (YOUR ORIGINAL BLOCK)
            if self.enable_cache:
                timer.start("cache_prediction")
                prepared_tensor = self._prepare_image(all_table_images[i])
                self._cache_prediction(
                    predict_details=predict_details,
                    doc_id=meta[i]["doc_id"],
                    page_idx=meta[i]["page_idx"],
                    table_idx=meta[i]["table_idx"],
                    table_image=all_table_images[i],
                    prepared_tensor=prepared_tensor,
                    orig_table_bbox=None,  # Not available in this flow
                    table_bbox=all_scaled_bboxes[i],
                    scale_factor=all_scale_factors[i]
                )
                timer.end("cache_prediction")

            multi_tf_output.append({
                "tf_responses": tf_responses,
                "predict_details": predict_details,
            })
        timer.end("phase3_package_outputs")

        # Print comprehensive timing summary
        from table_timing_debug import print_timing_summary
        print_timing_summary()

        return multi_tf_output
