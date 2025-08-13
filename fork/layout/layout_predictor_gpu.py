# Copyright IBM Corp. 2024 - 2025
# SPDX-License-Identifier: MIT

import logging
import os
import threading
from collections.abc import Iterable
from typing import Dict, List, Set, Union, Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from transformers import AutoModelForObjectDetection, RTDetrImageProcessor

from docling_ibm_models.layoutmodel.labels import LayoutLabels
from fork.timers import _CPUTimer, _CudaTimer
from fork.layout.gpu_preprocess import GPUPreprocessor, GPUPreprocessorV2

_log = logging.getLogger(__name__)
_model_init_lock = threading.Lock()

# ---- Static shape constants (graph-friendly) ----
FIXED_BS = int(os.getenv("LAYOUT_FIXED_BS", "64"))
FIXED_H, FIXED_W = 640, 640

# Recommended allocator setting (set in the environment before import/process start):
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


class LayoutPredictor:
    """
    Document layout prediction with GPU-accelerated preprocessing,
    Inductor compile (dynamic=False), and CUDA Graph replay on fixed shapes.
    """

    def __init__(
        self,
        artifact_path: str,
        device: str = "cuda",
        num_threads: int = 4,
        base_threshold: float = 0.3,
        blacklist_classes: Optional[Set[str]] = None,
        use_gpu_preprocess: bool = True,
        gpu_preprocess_version: int = 2,  # 1 or 2
    ):
        self._black_classes = blacklist_classes if blacklist_classes is not None else set()
        self._labels = LayoutLabels()
        self._threshold = base_threshold
        self._compat_mode = os.getenv("DOCLING_GPU_COMPAT_MODE", "").lower() in ("1", "true", "yes")

        self._device = torch.device(device)
        self._num_threads = num_threads
        if self._device.type == "cpu":
            torch.set_num_threads(self._num_threads)

        # ---- Load configs & processor ----
        self._processor_config = os.path.join(artifact_path, "preprocessor_config.json")
        self._model_config = os.path.join(artifact_path, "config.json")
        self._st_fn = os.path.join(artifact_path, "model.safetensors")
        if not os.path.isfile(self._st_fn):
            raise FileNotFoundError(f"Missing safe tensors file: {self._st_fn}")
        if not os.path.isfile(self._processor_config):
            raise FileNotFoundError(f"Missing processor config file: {self._processor_config}")
        if not os.path.isfile(self._model_config):
            raise FileNotFoundError(f"Missing model config file: {self._model_config}")

        self._image_preprocessor = RTDetrImageProcessor.from_json_file(self._processor_config)
        self._image_postprocessor = RTDetrImageProcessor.from_json_file(self._processor_config)

        # ---- Optional GPU preprocessor (must emit 640x640) ----
        self._use_gpu_preprocess = use_gpu_preprocess and self._device.type == "cuda"
        self._gpu_preprocessor = None

        if self._use_gpu_preprocess:
            Preproc = GPUPreprocessorV2 if gpu_preprocess_version == 2 else GPUPreprocessor
            self._gpu_preprocessor = Preproc(
                size={"height": FIXED_H, "width": FIXED_W},
                do_pad=True,
                pad_size={"height": FIXED_H, "width": FIXED_W},
                do_rescale=self._image_preprocessor.do_rescale,
                rescale_factor=self._image_preprocessor.rescale_factor,
                do_normalize=self._image_preprocessor.do_normalize,
                mean=tuple(self._image_preprocessor.image_mean),
                std=tuple(self._image_preprocessor.image_std),
                device=str(self._device),
                dtype=torch.float32,
            )
            _log.info(f"Using GPU preprocessor v{gpu_preprocess_version}")

        # ---- Model load + compile ----
        with _model_init_lock:
            self._model = AutoModelForObjectDetection.from_pretrained(
                artifact_path, config=self._model_config, device_map=self._device
            )
            self._model.eval()

            if self._device.type == "cuda":
                # Stable, fast math on Ampere+ (no accuracy cliff for this use-case)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                # You observed benchmark=True was slower; keep it False to avoid search overhead.
                torch.backends.cudnn.benchmark = False

                # Avoid per-batch layout conversions
                self._model.to(memory_format=torch.channels_last)

                # Inductor compile settings
                from torch._inductor import config as ind_cfg
                ind_cfg.triton.cudagraphs = True          # enable graph capture
                ind_cfg.max_autotune_gemm = False         # stable; re-test later if desired
                ind_cfg.coordinate_descent_tuning = False

                self._model = torch.compile(
                    self._model,
                    dynamic=False,          # STATIC shapes/strides/dtypes
                    fullgraph=True,         # avoid graph breaks
                    mode="reduce-overhead"  # low-latency small graphs
                )

            # ---- Persistent device input buffer (stable address for graphs) ----
            if self._device.type == "cuda":
                self._static_in = torch.empty(
                    FIXED_BS, 3, FIXED_H, FIXED_W, device=self._device,
                    memory_format=torch.channels_last, dtype=torch.float32
                )

                # ---- Warm-up: 1st = compile, 2nd = graph capture ----
                with torch.inference_mode():
                    for _ in range(2):
                        _ = self._model(pixel_values=self._static_in)
                torch.cuda.synchronize()

        # ---- Class map / offsets ----
        self._model_name = type(self._model).__name__
        if self._model_name == "RTDetrForObjectDetection":
            self._classes_map = self._labels.shifted_canonical_categories()
            self._label_offset = 1
        else:
            self._classes_map = self._labels.canonical_categories()
            self._label_offset = 0

        _log.debug("LayoutPredictor settings: %s", self.info())

    # ----------------- internals -----------------

    def _create_timer(self):
        return _CudaTimer() if self._device.type == "cuda" else _CPUTimer()

    def _store_timings(self, timer):
        self._t_preprocess_ms = timer.get_time('preprocess')
        self._t_predict_ms = timer.get_time('predict')
        self._t_postprocess_ms = timer.get_time('postprocess')

    def _stable_sort_result(self, res: Dict[str, Tensor]) -> Dict[str, Tensor]:
        boxes, scores, labels = res["boxes"], res["scores"], res["labels"]
        key = torch.stack([
            labels.to(torch.int64),
            (-scores).to(torch.float32),
            boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3],
        ], dim=1)
        idx = torch.arange(key.size(0), device=key.device)
        for col in range(key.size(1) - 1, -1, -1):
            vals = key[idx, col]
            order = torch.argsort(vals, stable=True)
            idx = idx[order]
        return {"boxes": boxes[idx], "scores": scores[idx], "labels": labels[idx]}

    def info(self) -> dict:
        return {
            "model_name": self._model_name,
            "safe_tensors_file": self._st_fn,
            "device": self._device.type,
            "num_threads": self._num_threads,
            "image_size": {"height": FIXED_H, "width": FIXED_W},
            "threshold": self._threshold,
            "use_gpu_preprocess": self._use_gpu_preprocess,
            "fixed_batch": FIXED_BS,
        }

    # ----------- hot path (fixed shape + graphs) -----------

    @torch.inference_mode()
    def _preprocess_batch(self, pil_images: List[Image.Image]) -> torch.Tensor:
        """
        Returns CPU tensor [B,3,640,640], channels_last, pinned for async H2D.
        Pads or truncates to FIXED_BS by design (caller ensures size).
        """
        if self._use_gpu_preprocess and self._gpu_preprocessor is not None:
            # GPU preprocessor emits device tensor; we still copy into self._static_in (stable address).
            pre = self._gpu_preprocessor.preprocess_batch(pil_images)
            pixel_values = pre["pixel_values"]  # [B,3,640,640] on device
            # Ensure channels_last; then copy into persistent buffer
            pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)
            self._static_in.copy_(pixel_values, non_blocking=True)
            return self._static_in  # already on device, correct address
        else:
            # CPU path via HF processor; create pinned tensor for async H2D
            inputs = self._image_preprocessor(images=pil_images, return_tensors="pt")
            pixel_values = inputs.pixel_values  # [B,3,640,640] on CPU
            # pin + channels_last + async H2D into persistent buffer
            pixel_values = pixel_values.pin_memory()
            # Copy into persistent buffer, maintaining stable device address
            self._static_in.copy_(
                pixel_values.to(dtype=self._static_in.dtype, non_blocking=True),
                non_blocking=True
            )
            return self._static_in

    def _to_pil_rgb(self, images: List[Union[Image.Image, np.ndarray]]) -> List[Image.Image]:
        out = []
        for img in images:
            if isinstance(img, Image.Image):
                out.append(img.convert("RGB"))
            elif isinstance(img, np.ndarray):
                out.append(Image.fromarray(img).convert("RGB"))
            else:
                raise TypeError("Unsupported input image format")
        return out

    def _postprocess(
        self,
        outputs,
        target_sizes: torch.Tensor,
    ) -> List[Dict[str, Tensor]]:
        threshold = max(0.0, self._threshold - 1e-4) if self._compat_mode else self._threshold
        res = self._image_postprocessor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )
        if self._compat_mode:
            res = [self._stable_sort_result(r) for r in res]
        # Move to CPU in bulk to avoid later syncs
        out = []
        for r in res:
            out.append({
                "boxes": r["boxes"].detach().to("cpu"),
                "scores": r["scores"].detach().to("cpu"),
                "labels": r["labels"].detach().to("cpu"),
            })
        return out

    # ----------------- public API -----------------

    @torch.inference_mode()
    def predict_batch(self, images: List[Union[Image.Image, np.ndarray]]) -> List[List[dict]]:
        """
        Runs in fixed-size chunks of 64 to preserve CUDA Graph replay.
        Returns list of predictions per original image (order preserved).
        """
        if not images:
            return []

        # Convert all to PIL RGB once
        pil_all = self._to_pil_rgb(images)
        N = len(pil_all)
        results_all: List[List[dict]] = []

        # Process in chunks of FIXED_BS
        for start in range(0, N, FIXED_BS):
            end = min(start + FIXED_BS, N)
            chunk = pil_all[start:end]
            b = len(chunk)

            # Pad chunk to FIXED_BS by repeating the last image
            if b < FIXED_BS:
                chunk = chunk + [chunk[-1]] * (FIXED_BS - b)

            timer = self._create_timer()

            with timer.time_section('preprocess'):
                # Prepare static device input with stable address/shape
                device_input = self._preprocess_batch(chunk)
                # Sanity: enforce memory format + shape
                assert device_input.shape == (FIXED_BS, 3, FIXED_H, FIXED_W)
                assert device_input.is_contiguous(memory_format=torch.channels_last)

            with timer.time_section('predict'):
                outputs = self._model(pixel_values=device_input)

            # target_sizes is (H, W) per *real* image
            with timer.time_section('postprocess'):
                target_sizes = torch.tensor([img.size[::-1] for img in chunk[:b]], dtype=torch.long)
                res_cpu = self._postprocess(outputs, target_sizes=target_sizes)

            timer.finalize()
            self._store_timings(timer)

            # Convert to client format and trim padding
            for img, res in zip(chunk[:b], res_cpu[:b]):
                w, h = img.size
                preds = []
                for score, label_id, box in zip(res["scores"], res["labels"], res["boxes"]):
                    s = float(score.item())
                    lid = int(label_id.item()) + self._label_offset
                    lbl = self._classes_map[lid]
                    if lbl in self._black_classes:
                        continue
                    x1, y1, x2, y2 = [float(bi.item()) for bi in box]
                    l = min(w, max(0.0, x1)); t = min(h, max(0.0, y1))
                    r = min(w, max(0.0, x2)); btm = min(h, max(0.0, y2))
                    preds.append({"l": l, "t": t, "r": r, "b": btm, "label": lbl, "confidence": s})
                results_all.append(preds)

        return results_all

    @torch.inference_mode()
    def predict(self, orig_img: Union[Image.Image, np.ndarray]) -> Iterable[dict]:
        res = self.predict_batch([orig_img])
        if res:
            for pred in res[0]:
                yield pred

# #
# # Copyright IBM Corp. 2024 - 2024
# # SPDX-License-Identifier: MIT
# #
# """
# GPU-accelerated layout predictor with optional GPU preprocessing.
# This is a drop-in replacement for LayoutPredictor with GPU preprocessing support.
# """
#
# import logging
# import os
# import threading
# from collections.abc import Iterable
# from typing import Dict, List, Set, Union, Optional
#
# import numpy as np
# import torch
# from PIL import Image
# from torch import Tensor
# from transformers import AutoModelForObjectDetection, RTDetrImageProcessor
#
# from docling_ibm_models.layoutmodel.labels import LayoutLabels
# from fork.timers import _CPUTimer, _CudaTimer
# from fork.layout.gpu_preprocess import GPUPreprocessor, GPUPreprocessorV2
#
# _log = logging.getLogger(__name__)
#
# # Fixed batch size for CUDA graphs
# FIXED_BS = 64
#
# # Global lock for model initialization to prevent threading issues
# _model_init_lock = threading.Lock()
#
#
# class LayoutPredictor:
#     """
#     Document layout prediction with GPU-accelerated preprocessing.
#     """
#
#     def __init__(
#         self,
#         artifact_path: str,
#         device: str = "cpu",
#         num_threads: int = 4,
#         base_threshold: float = 0.3,
#         blacklist_classes: Optional[Set[str]] = None,
#         use_gpu_preprocess: bool = True,
#         gpu_preprocess_version: int = 1,  # 1 or 2
#     ):
#         """w
#         Initialize layout predictor with optional GPU preprocessing.
#
#         Parameters
#         ----------
#         artifact_path: Path for the model torch file.
#         device: (Optional) device to run the inference.
#         num_threads: (Optional) Number of threads to run the inference if device = 'cpu'
#         use_gpu_preprocess: (Optional) Whether to use GPU-accelerated preprocessing
#         gpu_preprocess_version: (Optional) Which GPU preprocessor version to use (1 or 2)
#
#         Raises
#         ------
#         FileNotFoundError when the model's torch file is missing
#         """
#         # Blacklisted classes
#         self._black_classes = blacklist_classes if blacklist_classes is not None else set()
#
#         # Canonical classes
#         self._labels = LayoutLabels()
#
#         # Set basic params
#         self._threshold = base_threshold
#
#         # Enable compatibility mode if environment variable is set
#         self._compat_mode = os.getenv("DOCLING_GPU_COMPAT_MODE", "").lower() in ("1", "true", "yes")
#
#         # Set number of threads for CPU
#         self._device = torch.device(device)
#         self._num_threads = num_threads
#         if device == "cpu":
#             torch.set_num_threads(self._num_threads)
#
#         # Load model file and configurations
#         self._processor_config = os.path.join(artifact_path, "preprocessor_config.json")
#         self._model_config = os.path.join(artifact_path, "config.json")
#         self._st_fn = os.path.join(artifact_path, "model.safetensors")
#         if not os.path.isfile(self._st_fn):
#             raise FileNotFoundError("Missing safe tensors file: {}".format(self._st_fn))
#         if not os.path.isfile(self._processor_config):
#             raise FileNotFoundError(
#                 f"Missing processor config file: {self._processor_config}"
#             )
#         if not os.path.isfile(self._model_config):
#             raise FileNotFoundError(f"Missing model config file: {self._model_config}")
#
#         # Load HF image processor for fallback/comparison
#         self._image_preprocessor = RTDetrImageProcessor.from_json_file(
#             self._processor_config
#         )
#         self._image_postprocessor = RTDetrImageProcessor.from_json_file(
#             self._processor_config
#         )
#
#         # Initialize GPU preprocessor if requested
#         self._use_gpu_preprocess = use_gpu_preprocess and device != "cpu"
#         self._gpu_preprocessor = None
#
#         if self._use_gpu_preprocess:
#             # Extract size configuration from HF processor
#             size_config = self._image_preprocessor.size
#
#             # Choose GPU preprocessor version
#             PreprocessorClass = GPUPreprocessorV2 if gpu_preprocess_version == 2 else GPUPreprocessor
#
#             self._gpu_preprocessor = PreprocessorClass(
#                 size=size_config,
#                 do_pad=self._image_preprocessor.do_pad,
#                 pad_size=self._image_preprocessor.pad_size,
#                 do_rescale=self._image_preprocessor.do_rescale,
#                 rescale_factor=self._image_preprocessor.rescale_factor,
#                 do_normalize=self._image_preprocessor.do_normalize,
#                 mean=tuple(self._image_preprocessor.image_mean),
#                 std=tuple(self._image_preprocessor.image_std),
#                 device=str(self._device),  # Keep full device string (e.g., "cuda:1")
#                 dtype=torch.float32,  # Use float32 for stability
#             )
#             _log.info(f"Using GPU preprocessor v{gpu_preprocess_version}")
#
#         # Use lock to prevent threading issues during model initialization
#         with _model_init_lock:
#             self._model = AutoModelForObjectDetection.from_pretrained(
#                 artifact_path, config=self._model_config, device_map=self._device
#             )
#             self._model.eval()
#
#             # Step 0: TF32 + channels_last optimization for CUDA
#             if self._device.type == "cuda":
#                 # Fast default math for Ampere+ without changing numerics materially
#                 torch.backends.cuda.matmul.allow_tf32 = True
#                 torch.backends.cudnn.allow_tf32 = True
#                 torch.backends.cudnn.benchmark = False  # <-- disable benchmark for varying shapes
#
#                 # Avoid per-batch layout conversions
#                 self._model.to(memory_format=torch.channels_last)
#
#                 # ----- Configure torch.compile for CUDA graphs -----
#                 from torch._inductor import config as ind_cfg
#                 ind_cfg.triton.cudagraphs = True      # allow graph capture
#                 ind_cfg.max_autotune_gemm = False     # stable; re-enable later to test
#                 ind_cfg.coordinate_descent_tuning = False
#
#                 # Compile with static shapes for CUDA graph capture
#                 self._model = torch.compile(
#                     self._model,
#                     dynamic=False,        # shapes fixed = good
#                     fullgraph=True,       # avoid graph breaks
#                     mode="reduce-overhead"
#                 )
#
#                 # Warmup: compile + graph capture
#                 _log.info(f"Warming up model compilation for fixed batch size {FIXED_BS}...")
#                 with torch.inference_mode():
#                     dummy = torch.empty(FIXED_BS, 3, 640, 640, device=self._device,
#                                       memory_format=torch.channels_last, dtype=torch.float32)
#                     for i in range(2):     # first = compile, second = graph capture
#                         _ = self._model(pixel_values=dummy)
#                         if i == 0:
#                             _log.info("✓ Model compilation complete")
#                     torch.cuda.synchronize()
#                     _log.info("✓ CUDA graph capture complete")
#
#                 _log.info("Model compiled with Inductor + CUDA graphs (fixed batch=64)")
#
#         # Set classes map
#         self._model_name = type(self._model).__name__
#         if self._model_name == "RTDetrForObjectDetection":
#             self._classes_map = self._labels.shifted_canonical_categories()
#             self._label_offset = 1
#         else:
#             self._classes_map = self._labels.canonical_categories()
#             self._label_offset = 0
#
#         _log.debug("LayoutPredictorGPU settings: {}".format(self.info()))
#
#
#     def _stable_sort_result(self, res: Dict[str, Tensor]) -> Dict[str, Tensor]:
#         """
#         Stable sort detection result to make order deterministic.
#         Sort by: label, -score, x1, y1, x2, y2 (consistent and stable).
#         """
#         boxes, scores, labels = res["boxes"], res["scores"], res["labels"]
#
#         # Create sort key: [label, -score, x1, y1, x2, y2]
#         key = torch.stack([
#             labels.to(torch.int64),
#             (-scores).to(torch.float32),
#             boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
#         ], dim=1)
#
#         # Lexicographic sort: sort by last column first, then work backwards
#         idx = torch.arange(key.size(0), device=key.device)
#         for col in range(key.size(1) - 1, -1, -1):  # last -> first
#             vals = key[idx, col]  # Use current permutation
#             order = torch.argsort(vals, stable=True)
#             idx = idx[order]
#
#         return {
#             "boxes": boxes[idx],
#             "scores": scores[idx],
#             "labels": labels[idx],
#         }
#
#     def _create_timer(self):
#         """Create a timer appropriate for the device."""
#         if self._device.type == "cuda":
#             return _CudaTimer()
#         else:
#             return _CPUTimer()
#
#     def _store_timings(self, timer):
#         """Store timing results from timer."""
#         self._t_preprocess_ms = timer.get_time('preprocess')
#         self._t_predict_ms = timer.get_time('predict')
#         self._t_postprocess_ms = timer.get_time('postprocess')
#
#     def info(self) -> dict:
#         """
#         Get information about the configuration of LayoutPredictor
#         """
#         info = {
#             "model_name": self._model_name,
#             "safe_tensors_file": self._st_fn,
#             "device": self._device.type,
#             "num_threads": self._num_threads,
#             "image_size": self._image_preprocessor.size,
#             "threshold": self._threshold,
#             "use_gpu_preprocess": self._use_gpu_preprocess,
#         }
#         return info
#
#     @torch.inference_mode()
#     def predict_batch(
#             self, images: List[Union[Image.Image, np.ndarray]]
#     ) -> List[List[dict]]:
#         """
#         Batch prediction for multiple images with optional GPU preprocessing.
#
#         Parameters
#         ----------
#         images : List[Union[Image.Image, np.ndarray]]
#             List of images to process in a single batch
#
#         Returns
#         -------
#         List[List[dict]]
#             List of prediction lists, one per input image. Each prediction dict contains:
#             "label", "confidence", "l", "t", "r", "b"
#         """
#         if not images:
#             return []
#
#         # Always pad to fixed batch size for CUDA graphs
#         original_count = len(images)
#         if original_count > FIXED_BS:
#             raise ValueError(f"Batch too large ({original_count}) for FIXED_BS={FIXED_BS}")
#
#         # Pad to fixed batch size
#         if original_count < FIXED_BS:
#             padding_needed = FIXED_BS - original_count
#             images = images + [images[-1]] * padding_needed
#
#         # Convert all images to RGB PIL format for consistency
#         pil_images = []
#         for img in images:
#             if isinstance(img, Image.Image):
#                 pil_images.append(img.convert("RGB"))
#             elif isinstance(img, np.ndarray):
#                 pil_images.append(Image.fromarray(img).convert("RGB"))
#             else:
#                 raise TypeError("Not supported input image format")
#
#         timer = self._create_timer()
#
#         # Get target sizes for all images
#         target_sizes = torch.tensor([img.size[::-1] for img in pil_images])
#
#         if self._use_gpu_preprocess and self._gpu_preprocessor is not None:
#             # GPU preprocessing path
#             with timer.time_section('preprocess'):
#                 preprocessed = self._gpu_preprocessor.preprocess_batch(pil_images)
#                 pixel_values = preprocessed['pixel_values']
#                 if pixel_values.device != self._device:
#                     pixel_values = pixel_values.to(self._device, non_blocking=True)
#                 pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)
#         else:
#             # CPU preprocessing path (original HF)
#             with timer.time_section('preprocess'):
#                 inputs = self._image_preprocessor(images=pil_images, return_tensors="pt")
#                 pixel_values = inputs.pixel_values
#
#                 # Ensure proper memory layout for CUDA graphs
#                 if self._device.type == "cuda":
#                     if pixel_values.device.type == "cpu":
#                         pixel_values = pixel_values.pin_memory()
#                     pixel_values = pixel_values.to(self._device, non_blocking=True)
#                     pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)
#
#                     # Ensure fixed shape for CUDA graphs
#                     assert pixel_values.shape == (FIXED_BS, 3, 640, 640), f"Expected shape ({FIXED_BS}, 3, 640, 640), got {pixel_values.shape}"
#                     assert pixel_values.is_contiguous(memory_format=torch.channels_last), "Tensor must be channels_last contiguous"
#                 else:
#                     pixel_values = pixel_values.to(self._device)
#
#         with timer.time_section('predict'):
#             outputs = self._model(pixel_values=pixel_values)
#
#         with timer.time_section('postprocess'):
#             # Apply threshold hysteresis in compatibility mode
#             threshold = self._threshold
#             if self._compat_mode:
#                 threshold = max(0.0, threshold - 1e-4)
#
#             results_list: List[Dict[str, Tensor]] = (
#                 self._image_postprocessor.post_process_object_detection(
#                     outputs,
#                     target_sizes=target_sizes,
#                     threshold=threshold,
#                 )
#             )
#
#             # Apply stable sorting in compatibility mode
#             if self._compat_mode:
#                 results_list = [self._stable_sort_result(r) for r in results_list]
#
#             # Move tensors to CPU in bulk to avoid hidden syncs later
#             results_cpu = []
#             for r in results_list:
#                 results_cpu.append({
#                     "boxes": r["boxes"].detach().to("cpu"),
#                     "scores": r["scores"].detach().to("cpu"),
#                     "labels": r["labels"].detach().to("cpu"),
#                 })
#
#         timer.finalize()
#         self._store_timings(timer)
#
#         # Convert results to standard format for each image
#         all_predictions = []
#
#         for img, results in zip(pil_images, results_cpu):
#             w, h = img.size
#             predictions = []
#
#             for score, label_id, box in zip(
#                     results["scores"], results["labels"], results["boxes"]
#             ):
#                 score = float(score.item())
#                 label_id = int(label_id.item()) + self._label_offset
#                 label_str = self._classes_map[label_id]
#
#                 # Filter out blacklisted classes
#                 if label_str in self._black_classes:
#                     continue
#
#                 bbox_float = [float(b.item()) for b in box]
#                 l = min(w, max(0, bbox_float[0]))
#                 t = min(h, max(0, bbox_float[1]))
#                 r = min(w, max(0, bbox_float[2]))
#                 b = min(h, max(0, bbox_float[3]))
#
#                 predictions.append(
#                     {
#                         "l": l,
#                         "t": t,
#                         "r": r,
#                         "b": b,
#                         "label": label_str,
#                         "confidence": score,
#                     }
#                 )
#
#             all_predictions.append(predictions)
#
#         # Return only original results (trim padding)
#         return all_predictions[:original_count]
#
#     @torch.inference_mode()
#     def predict(self, orig_img: Union[Image.Image, np.ndarray]) -> Iterable[dict]:
#         """
#         Predict bounding boxes for a given image (single image version for compatibility).
#         """
#         # Use batch prediction internally
#         results = self.predict_batch([orig_img])
#         if results:
#             for pred in results[0]:
#                 yield pred
