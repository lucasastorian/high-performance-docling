#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import logging
import os
import threading
import time
from collections.abc import Iterable
from typing import Dict, List, Set, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from transformers import AutoModelForObjectDetection, RTDetrImageProcessor

from docling_ibm_models.layoutmodel.labels import LayoutLabels
from fork.layout.image_processing_rt_detr import OptimizedRTDetrImageProcessor

_log = logging.getLogger(__name__)

# Global lock for model initialization to prevent threading issues
_model_init_lock = threading.Lock()


class _CudaTimer:
    """CUDA event-based timer for GPU operations."""
    
    def __init__(self):
        self.events = {}
        self.times = {}
    
    def time_section(self, name: str):
        """Context manager for timing a section."""
        return _CudaTimerContext(self, name)
    
    def start_section(self, name: str):
        """Start timing a section."""
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        self.events[name] = (start_event, end_event)
    
    def end_section(self, name: str):
        """End timing a section."""
        if name in self.events:
            start_event, end_event = self.events[name]
            end_event.record()
    
    def finalize(self):
        """Synchronize and compute all timings."""
        # Single sync for all events
        if self.events:
            last_event = None
            for start_event, end_event in self.events.values():
                last_event = end_event
            if last_event:
                last_event.synchronize()
        
        # Compute all elapsed times
        for name, (start_event, end_event) in self.events.items():
            self.times[name] = start_event.elapsed_time(end_event)
    
    def get_time(self, name: str) -> float:
        """Get timing for a section in milliseconds."""
        return self.times.get(name, 0.0)


class _CudaTimerContext:
    """Context manager for CUDA timing sections."""
    
    def __init__(self, timer: _CudaTimer, name: str):
        self.timer = timer
        self.name = name
    
    def __enter__(self):
        self.timer.start_section(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.end_section(self.name)


class _CPUTimer:
    """CPU-based timer for fallback."""
    
    def __init__(self):
        self.times = {}
        self.start_times = {}
    
    def time_section(self, name: str):
        """Context manager for timing a section."""
        return _CPUTimerContext(self, name)
    
    def start_section(self, name: str):
        """Start timing a section."""
        self.start_times[name] = time.perf_counter()
    
    def end_section(self, name: str):
        """End timing a section."""
        if name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[name]
            self.times[name] = elapsed * 1000  # Convert to ms
    
    def finalize(self):
        """No-op for CPU timer."""
        pass
    
    def get_time(self, name: str) -> float:
        """Get timing for a section in milliseconds."""
        return self.times.get(name, 0.0)


class _CPUTimerContext:
    """Context manager for CPU timing sections."""
    
    def __init__(self, timer: _CPUTimer, name: str):
        self.timer = timer
        self.name = name
    
    def __enter__(self):
        self.timer.start_section(self.name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.end_section(self.name)


class LayoutPredictor:
    """
    Document layout prediction using safe tensors
    """

    def __init__(
        self,
        artifact_path: str,
        device: str = "cpu",
        num_threads: int = 4,
        base_threshold: float = 0.3,
        blacklist_classes: Set[str] = set(),
    ):
        """
        Provide the artifact path that contains the LayoutModel file

        Parameters
        ----------
        artifact_path: Path for the model torch file.
        device: (Optional) device to run the inference.
        num_threads: (Optional) Number of threads to run the inference if device = 'cpu'

        Raises
        ------
        FileNotFoundError when the model's torch file is missing
        """
        # Blacklisted classes
        self._black_classes = blacklist_classes  # set(["Form", "Key-Value Region"])

        # Canonical classes
        self._labels = LayoutLabels()

        # Set basic params
        self._threshold = base_threshold  # Score threshold

        # Set number of threads for CPU
        self._device = torch.device(device)
        self._num_threads = num_threads
        if device == "cpu":
            torch.set_num_threads(self._num_threads)

        # Load model file and configurations
        self._processor_config = os.path.join(artifact_path, "preprocessor_config.json")
        self._model_config = os.path.join(artifact_path, "config.json")
        self._st_fn = os.path.join(artifact_path, "model.safetensors")
        if not os.path.isfile(self._st_fn):
            raise FileNotFoundError("Missing safe tensors file: {}".format(self._st_fn))
        if not os.path.isfile(self._processor_config):
            raise FileNotFoundError(
                f"Missing processor config file: {self._processor_config}"
            )
        if not os.path.isfile(self._model_config):
            raise FileNotFoundError(f"Missing model config file: {self._model_config}")

        # Load model and move to device
        self._image_preprocessor = OptimizedRTDetrImageProcessor.from_json_file(
            self._processor_config
        )

        self._image_postprocessor = RTDetrImageProcessor.from_json_file(
            self._processor_config
        )

        # Use lock to prevent threading issues during model initialization
        with _model_init_lock:
            self._model = AutoModelForObjectDetection.from_pretrained(
                artifact_path, config=self._model_config, device_map=self._device
            )
            self._model.eval()

        # Set classes map
        self._model_name = type(self._model).__name__
        if self._model_name == "RTDetrForObjectDetection":
            self._classes_map = self._labels.shifted_canonical_categories()
            self._label_offset = 1
        else:
            self._classes_map = self._labels.canonical_categories()
            self._label_offset = 0

        _log.debug("LayoutPredictor settings: {}".format(self.info()))

    def _create_timer(self):
        """Create a timer appropriate for the device."""
        if self._device.type == "cuda":
            return _CudaTimer()
        else:
            return _CPUTimer()

    def _store_timings(self, timer):
        """Store timing results from timer."""
        self._t_preprocess_ms = timer.get_time('preprocess')
        self._t_predict_ms = timer.get_time('predict')
        self._t_postprocess_ms = timer.get_time('postprocess')

    def info(self) -> dict:
        """
        Get information about the configuration of LayoutPredictor
        """
        info = {
            "model_name": self._model_name,
            "safe_tensors_file": self._st_fn,
            "device": self._device.type,
            "num_threads": self._num_threads,
            "image_size": self._image_preprocessor.size,
            "threshold": self._threshold,
        }
        return info

    @torch.inference_mode()
    def predict(self, orig_img: Union[Image.Image, np.ndarray]) -> Iterable[dict]:
        """
        Predict bounding boxes for a given image.
        The origin (0, 0) is the top-left corner and the predicted bbox coords are provided as:
        [left, top, right, bottom]

        Parameter
        ---------
        origin_img: Image to be predicted as a PIL Image object or numpy array.

        Yield
        -----
        Bounding box as a dict with the keys: "label", "confidence", "l", "t", "r", "b"

        Raises
        ------
        TypeError when the input image is not supported
        """
        # Convert image format
        if isinstance(orig_img, Image.Image):
            page_img = orig_img.convert("RGB")
        elif isinstance(orig_img, np.ndarray):
            page_img = Image.fromarray(orig_img).convert("RGB")
        else:
            raise TypeError("Not supported input image format")

        target_sizes = torch.tensor([page_img.size[::-1]])
        inputs = self._image_preprocessor(images=[page_img], return_tensors="pt").to(
            self._device
        )
        outputs = self._model(**inputs)
        results: List[Dict[str, Tensor]] = (
            self._image_postprocessor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self._threshold,
            )
        )

        w, h = page_img.size
        result = results[0]
        for score, label_id, box in zip(
            result["scores"], result["labels"], result["boxes"]
        ):
            score = float(score.item())

            label_id = int(label_id.item()) + self._label_offset
            label_str = self._classes_map[label_id]

            # Filter out blacklisted classes
            if label_str in self._black_classes:
                continue

            bbox_float = [float(b.item()) for b in box]
            l = min(w, max(0, bbox_float[0]))
            t = min(h, max(0, bbox_float[1]))
            r = min(w, max(0, bbox_float[2]))
            b = min(h, max(0, bbox_float[3]))
            yield {
                "l": l,
                "t": t,
                "r": r,
                "b": b,
                "label": label_str,
                "confidence": score,
            }

    # @torch.inference_mode()
    # def predict_batch(
    #         self, images: List[Union[Image.Image, np.ndarray]]
    # ) -> List[List[dict]]:
    #     """
    #     Batch prediction for multiple images - more efficient than calling predict() multiple times.
    #
    #     Parameters
    #     ----------
    #     images : List[Union[Image.Image, np.ndarray]]
    #         List of images to process in a single batch
    #
    #     Returns
    #     -------
    #     List[List[dict]]
    #         List of prediction lists, one per input image. Each prediction dict contains:
    #         "label", "confidence", "l", "t", "r", "b"
    #     """
    #     if not images:
    #         return []
    #
    #     # Convert all images to RGB PIL format
    #     pil_images = []
    #     for img in images:
    #         if isinstance(img, Image.Image):
    #             pil_images.append(img.convert("RGB"))
    #         elif isinstance(img, np.ndarray):
    #             pil_images.append(Image.fromarray(img).convert("RGB"))
    #         else:
    #             raise TypeError("Not supported input image format")
    #
    #     # Get target sizes for all images
    #     target_sizes = torch.tensor([img.size[::-1] for img in pil_images])
    #
    #     # Process all images in a single batch
    #     inputs = self._image_preprocessor(images=pil_images, return_tensors="pt").to(
    #         self._device
    #     )
    #     outputs = self._model(**inputs)
    #
    #     # Post-process all results at once
    #     results_list: List[Dict[str, Tensor]] = (
    #         self._image_postprocessor.post_process_object_detection(
    #             outputs,
    #             target_sizes=target_sizes,
    #             threshold=self._threshold,
    #         )
    #     )
    #
    #     # Convert results to standard format for each image
    #     all_predictions = []
    #
    #     for img, results in zip(pil_images, results_list):
    #         w, h = img.size
    #         predictions = []
    #
    #         for score, label_id, box in zip(
    #                 results["scores"], results["labels"], results["boxes"]
    #         ):
    #             score = float(score.item())
    #             label_id = int(label_id.item()) + self._label_offset
    #             label_str = self._classes_map[label_id]
    #
    #             # Filter out blacklisted classes
    #             if label_str in self._black_classes:
    #                 continue
    #
    #             bbox_float = [float(b.item()) for b in box]
    #             l = min(w, max(0, bbox_float[0]))
    #             t = min(h, max(0, bbox_float[1]))
    #             r = min(w, max(0, bbox_float[2]))
    #             b = min(h, max(0, bbox_float[3]))
    #
    #             predictions.append(
    #                 {
    #                     "l": l,
    #                     "t": t,
    #                     "r": r,
    #                     "b": b,
    #                     "label": label_str,
    #                     "confidence": score,
    #                 }
    #             )
    #
    #         all_predictions.append(predictions)
    #
    #     return all_predictions

    @torch.inference_mode()
    def predict_batch(self, images: List[Union[Image.Image, np.ndarray]]) -> List[List[dict]]:
        if not images:
            return []

        # 1) Canonicalize inputs (RGB PIL) without branching later
        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            elif isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img).convert("RGB"))
            else:
                raise TypeError("Not supported input image format")

        # Timing setup
        timer = self._create_timer()

        # 2) Preprocess with the ORIGINAL HF processor (exact math preserved)
        #    NOTE: this builds CPU tensors; we'll pin & async-copy below.
        with timer.time_section('preprocess'):
            inputs = self._image_preprocessor(images=pil_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"]  # [B,3,H',W'] on CPU
            if self._device.type == "cuda":
                pixel_values = pixel_values.pin_memory().to(
                    self._device, non_blocking=True
                ).contiguous(memory_format=torch.channels_last)
            else:
                pixel_values = pixel_values.to(self._device)

        # 3) Forward pass with autocast (no output drift in object-detection heads)
        with timer.time_section('predict'):
            with torch.autocast(self._device.type, dtype=torch.float16 if self._device.type == "cuda" else torch.bfloat16,
                                enabled=self._device.type != "cpu"):
                outputs = self._model(pixel_values=pixel_values)

        # 4) Post-process on CPU in one go (HF logic retained)
        #    Build target_sizes as CPU tensor once (h, w per image).
        with timer.time_section('postprocess'):
            target_sizes = torch.tensor([im.size[::-1] for im in pil_images], dtype=torch.long)
            # HF helper expects model-device tensors; it handles device moves internally.
            results_list = self._image_postprocessor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self._threshold
            )

        # Finalize and store timing results
        timer.finalize()
        self._store_timings(timer)

        # 5) Bulk convert tensors -> Python dicts without .item() in loops
        all_predictions: List[List[dict]] = []
        for im, res in zip(pil_images, results_list):
            w, h = im.size
            # Move once
            boxes = res["boxes"].detach().cpu().numpy()  # [N,4]
            scores = res["scores"].detach().cpu().numpy()  # [N]
            labels = res["labels"].detach().cpu().numpy()  # [N]

            preds = []
            # (Optional) blacklist filtering on CPU – micro-fast in NumPy
            for box, score, lab in zip(boxes, scores, labels):
                lab = int(lab) + self._label_offset
                label_str = self._classes_map[lab]
                if label_str in self._black_classes:
                    continue
                l = float(min(w, max(0.0, box[0])))
                t = float(min(h, max(0.0, box[1])))
                r = float(min(w, max(0.0, box[2])))
                b = float(min(h, max(0.0, box[3])))
                preds.append({"l": l, "t": t, "r": r, "b": b, "label": label_str, "confidence": float(score)})

            all_predictions.append(preds)

        return all_predictions

