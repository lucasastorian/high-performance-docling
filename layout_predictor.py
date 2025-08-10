#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import time
import logging
import os
import io
import contextlib
import threading
from collections.abc import Iterable
from typing import Dict, List, Set, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoModelForObjectDetection, RTDetrImageProcessor

from docling_ibm_models.layoutmodel.labels import LayoutLabels

_log = logging.getLogger(__name__)

# Global lock for model initialization to prevent threading issues
_model_init_lock = threading.Lock()


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
            raise FileNotFoundError(f"Missing safe tensors file: {self._st_fn}")
        if not os.path.isfile(self._processor_config):
            raise FileNotFoundError(
                f"Missing processor config file: {self._processor_config}"
            )
        if not os.path.isfile(self._model_config):
            raise FileNotFoundError(f"Missing model config file: {self._model_config}")

        # Load model and move to device
        self._image_processor = RTDetrImageProcessor.from_json_file(
            self._processor_config
        )

        # Use lock to prevent threading issues during model initialization
        with _model_init_lock:
            self._model = AutoModelForObjectDetection.from_pretrained(
                artifact_path, config=self._model_config, device_map=self._device
            )
            self._model.eval()

        # Prefer channels_last memory format for better kernel perf (no numeric changes)
        try:
            self._model = self._model.to(memory_format=torch.channels_last)
        except Exception:
            # If a weird module refuses channels_last, skip silently
            pass

        # Set classes map
        self._model_name = type(self._model).__name__
        if self._model_name == "RTDetrForObjectDetection":
            self._classes_map = self._labels.shifted_canonical_categories()
            self._label_offset = 1
        else:
            self._classes_map = self._labels.canonical_categories()
            self._label_offset = 0

        _log.debug("LayoutPredictor settings: {}".format(self.info()))

    def info(self) -> dict:
        """
        Get information about the configuration of LayoutPredictor
        """
        info = {
            "model_name": self._model_name,
            "safe_tensors_file": self._st_fn,
            "device": self._device.type,
            "num_threads": self._num_threads,
            "image_size": self._image_processor.size,
            "threshold": self._threshold,
        }
        return info

    @torch.inference_mode()
    def predict(self, orig_img: Union[Image.Image, np.ndarray]) -> Iterable[dict]:
        """
        Predict bounding boxes for a given image.
        The origin (0, 0) is the top-left corner and the predicted bbox coords are provided as:
        [left, top, right, bottom]
        """
        # Convert image format
        if isinstance(orig_img, Image.Image):
            page_img = orig_img.convert("RGB")
        elif isinstance(orig_img, np.ndarray):
            page_img = Image.fromarray(orig_img).convert("RGB")
        else:
            raise TypeError("Not supported input image format")

        target_sizes = torch.tensor([page_img.size[::-1]])

        inputs = self._image_processor(images=[page_img], return_tensors="pt")
        pixel_values = inputs["pixel_values"]

        if self._device.type == "cuda":
            # contiguous channels_last + pinned host + non_blocking copy
            pixel_values = (
                pixel_values
                .contiguous(memory_format=torch.channels_last)
                .pin_memory()
                .to(self._device, non_blocking=True)
            )
        outputs = self._model(pixel_values=pixel_values)
        results: List[Dict[str, Tensor]] = (
            self._image_processor.post_process_object_detection(
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

    @torch.inference_mode()
    def predict_batch(
            self, images: List[Union[Image.Image, np.ndarray]]
    ) -> List[List[dict]]:
        """
        Batch prediction for multiple images - more efficient than calling predict() multiple times.

        Returns a list of per-image predictions.
        """
        if not images:
            return []

        # Convert all images to RGB PIL format
        t_convert0 = time.perf_counter()
        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            elif isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img).convert("RGB"))
            else:
                raise TypeError("Not supported input image format")
        t_convert1 = time.perf_counter()

        # Target sizes remain on CPU for postprocess
        target_sizes = torch.tensor([img.size[::-1] for img in pil_images])

        # Build inputs on CPU - this is the expensive operation!
        # Let's profile what the image processor is doing internally
        torch.cuda.synchronize() if self._device.type == "cuda" else None
        t0 = time.perf_counter()
        
        # The image processor likely does:
        # 1. Resize images to fixed size (expensive on CPU)
        # 2. Convert to tensors
        # 3. Normalize pixel values
        # Let's time this more granularly
        print(f"📊 Processing {len(pil_images)} images, sizes: {[img.size for img in pil_images[:3]]}...")
        
        # inputs = self._image_processor(images=pil_images, return_tensors="pt")
        t_pre = time.perf_counter()

        # Only move pixel_values; keep dict on CPU
        # pixel_values = inputs["pixel_values"]

        pixel_values = rtdetr_preprocess_tensor(pil_images, device=self._device.type)
        print(f"   Preprocessed tensor shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")

        assert pixel_values.shape[-2:] == (640, 640)

        # Prepare memory layout for better kernel perf without changing numerics
        t_layout0 = time.perf_counter()
        pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)
        t_layout1 = time.perf_counter()

        # # Move to device
        # if self._device.type == "cuda":
        #     t_pin0 = time.perf_counter()
        #     pixel_values = pixel_values.pin_memory()
        #     t_pin1 = time.perf_counter()
        #     pixel_values = pixel_values.to(self._device, non_blocking=True)
        #     torch.cuda.synchronize()
        #
        # elif self._device.type == "mps":
        #     pixel_values = pixel_values.to(self._device)
        # t_h2d = time.perf_counter()

        # Forward pass - autocast only benefits CUDA
        with torch.inference_mode():
            if self._device.type == "cuda":
                from torch import amp
                with amp.autocast("cuda", dtype=torch.float16):
                    outputs = self._model(pixel_values=pixel_values)
            else:
                outputs = self._model(pixel_values=pixel_values)
        t_fwd = time.perf_counter()

        # Post-process all results at once on CPU (unchanged numerics)
        results_list: List[Dict[str, Tensor]] = (
            self._image_processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=self._threshold,
            )
        )
        t_post = time.perf_counter()

        # Detailed timing breakdown
        print(
            f"⏱ Layout timing breakdown:\n"
            f"   img_convert: {t_convert1 - t_convert0:.3f}s\n"
            f"   preprocess: {t_pre - t0:.3f}s (RTDetrImageProcessor - resize/normalize)\n"
            f"   channels_last: {t_layout1 - t_layout0:.3f}s\n"
            f"   h2d: {t_h2d - t_layout1:.3f}s\n"
            f"   forward: {t_fwd - t_h2d:.3f}s\n"
            f"   post: {t_post - t_fwd:.3f}s"
        )
        
        if self._device.type == "cuda" and 't_pin0' in locals():
            print(f"   (pin_memory: {t_pin1 - t_pin0:.3f}s)")

        # Convert results to standard format for each image
        all_predictions: List[List[dict]] = []
        for img, results in zip(pil_images, results_list):
            w, h = img.size
            predictions = []
            for score, label_id, box in zip(
                    results["scores"], results["labels"], results["boxes"]
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

                predictions.append(
                    {
                        "l": l,
                        "t": t,
                        "r": r,
                        "b": b,
                        "label": label_str,
                        "confidence": score,
                    }
                )
            all_predictions.append(predictions)

        return all_predictions


def rtdetr_preprocess_tensor(pil_list, device="cuda"):
    # Convert PIL -> RGB np.uint8
    rgb_list = [np.array(im.convert("RGB"), dtype=np.uint8) for im in pil_list]
    t = torch.stack([torch.from_numpy(x) for x in rgb_list])          # [B,H,W,3], uint8
    t = t.permute(0, 3, 1, 2).contiguous().to(torch.float32)          # [B,3,H,W], float32
    t = t.div_(255.0)                                                 # rescale to [0,1]
    t = F.interpolate(t, size=(640, 640), mode="bilinear", align_corners=False)  # warp (no AR)
    t = t.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
    return t  # feed as pixel_values
