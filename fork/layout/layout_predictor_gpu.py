#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
"""
GPU-accelerated layout predictor with optional GPU preprocessing.
This is a drop-in replacement for LayoutPredictor with GPU preprocessing support.
"""

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

# Global lock for model initialization to prevent threading issues
_model_init_lock = threading.Lock()


class LayoutPredictor:
    """
    Document layout prediction with GPU-accelerated preprocessing.
    """

    def __init__(
        self,
        artifact_path: str,
        device: str = "cpu",
        num_threads: int = 4,
        base_threshold: float = 0.3,
        blacklist_classes: Optional[Set[str]] = None,
        use_gpu_preprocess: bool = True,
        gpu_preprocess_version: int = 1,  # 1 or 2
    ):
        """w
        Initialize layout predictor with optional GPU preprocessing.
        
        Parameters
        ----------
        artifact_path: Path for the model torch file.
        device: (Optional) device to run the inference.
        num_threads: (Optional) Number of threads to run the inference if device = 'cpu'
        use_gpu_preprocess: (Optional) Whether to use GPU-accelerated preprocessing
        gpu_preprocess_version: (Optional) Which GPU preprocessor version to use (1 or 2)
        
        Raises
        ------
        FileNotFoundError when the model's torch file is missing
        """
        # Blacklisted classes
        self._black_classes = blacklist_classes if blacklist_classes is not None else set()

        # Canonical classes
        self._labels = LayoutLabels()

        # Set basic params
        self._threshold = base_threshold
        
        # Enable compatibility mode if environment variable is set
        self._compat_mode = os.getenv("DOCLING_GPU_COMPAT_MODE", "").lower() in ("1", "true", "yes")

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

        # Load HF image processor for fallback/comparison
        self._image_preprocessor = RTDetrImageProcessor.from_json_file(
            self._processor_config
        )
        self._image_postprocessor = RTDetrImageProcessor.from_json_file(
            self._processor_config
        )

        # Initialize GPU preprocessor if requested
        self._use_gpu_preprocess = use_gpu_preprocess and device != "cpu"
        self._gpu_preprocessor = None
        
        if self._use_gpu_preprocess:
            # Extract size configuration from HF processor
            size_config = self._image_preprocessor.size
            
            # Choose GPU preprocessor version
            PreprocessorClass = GPUPreprocessorV2 if gpu_preprocess_version == 2 else GPUPreprocessor
            
            self._gpu_preprocessor = PreprocessorClass(
                size=size_config,
                do_pad=self._image_preprocessor.do_pad,
                pad_size=self._image_preprocessor.pad_size,
                do_rescale=self._image_preprocessor.do_rescale,
                rescale_factor=self._image_preprocessor.rescale_factor,
                do_normalize=self._image_preprocessor.do_normalize,
                mean=tuple(self._image_preprocessor.image_mean),
                std=tuple(self._image_preprocessor.image_std),
                device=str(self._device),  # Keep full device string (e.g., "cuda:1")
                dtype=torch.float32,  # Use float32 for stability
            )
            _log.info(f"Using GPU preprocessor v{gpu_preprocess_version}")

        # Use lock to prevent threading issues during model initialization
        with _model_init_lock:
            self._model = AutoModelForObjectDetection.from_pretrained(
                artifact_path, config=self._model_config, device_map=self._device
            )
            self._model.eval()
            
            # Step 0: TF32 + channels_last optimization for CUDA
            if self._device.type == "cuda":
                # Fast default math for Ampere+ without changing numerics materially
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = False  # <-- disable benchmark for varying shapes
                
                # Avoid per-batch layout conversions
                self._model.to(memory_format=torch.channels_last)
                
                # ----- Torch-TRT via torch.compile (env gated) -----
                self._use_torch_compile_trt = os.getenv("USE_TRT", "").lower() in ("1", "true", "yes")
                
                # Keep a fallback handle to the eager model
                self._pt_model = self._model
                
                if self._use_torch_compile_trt:
                    try:
                        # Turn on debug logs to see failing ops
                        os.environ["TORCH_TRT_LOG_LEVEL"] = "DEBUG"
                        os.environ["TORCH_LOGS"] = "+dynamo,graph_breaks"
                        os.environ["TORCH_COMPILE_DEBUG"] = "1"
                        
                        import torch_tensorrt  # ensure installed, version-matched to torch
                        
                        # Use FP16, not FP32 for better TRT coverage
                        self._trt_dtype = torch.float16
                        
                        # Backend options (fixes for empty engine issues)
                        trt_opts = {
                            "precision": self._trt_dtype,
                            "use_fast_partitioner": True,
                            "min_block_size": 1,                 # allow small clusters
                            "workspace_size": 6 << 30,           # bump to 6GB for big batches
                            "require_full_compilation": False,
                            "truncate_long_and_double": True,
                        }
                        
                        # Cast model weights to FP16 to match input dtype
                        self._model = self._model.to(dtype=self._trt_dtype)
                        
                        # Wrap the model. kwargs preserved (e.g., pixel_values=...)
                        self._model = torch.compile(self._model, backend="tensorrt", options=trt_opts)
                        
                        # Optional warmup to prebuild engines for common batch sizes
                        warm_batches = [1, 32, 64]
                        _log.info(f"Warming up TensorRT compilation for batch sizes: {warm_batches}")
                        for b in warm_batches:
                            try:
                                dummy = torch.zeros((b, 3, 640, 640), device=self._device, dtype=self._trt_dtype)
                                dummy = dummy.contiguous(memory_format=torch.channels_last)
                                with torch.inference_mode():
                                    _ = self._model(pixel_values=dummy)
                                _log.info(f"✓ TRT warmup complete for batch {b}")
                            except Exception as e:
                                _log.warning(f"TRT warmup failed for batch {b}: {e}")
                        
                        _log.info(f"TensorRT compilation enabled (precision={self._trt_dtype})")
                    except ImportError:
                        _log.warning("torch_tensorrt not available, falling back to PyTorch")
                        self._use_torch_compile_trt = False
                    except Exception as e:
                        _log.warning(f"TensorRT compilation failed: {e}, falling back to PyTorch")
                        self._use_torch_compile_trt = False
            else:
                self._use_torch_compile_trt = False

        # Set classes map
        self._model_name = type(self._model).__name__
        if self._model_name == "RTDetrForObjectDetection":
            self._classes_map = self._labels.shifted_canonical_categories()
            self._label_offset = 1
        else:
            self._classes_map = self._labels.canonical_categories()
            self._label_offset = 0

        _log.debug("LayoutPredictorGPU settings: {}".format(self.info()))


    def _stable_sort_result(self, res: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Stable sort detection result to make order deterministic.
        Sort by: label, -score, x1, y1, x2, y2 (consistent and stable).
        """
        boxes, scores, labels = res["boxes"], res["scores"], res["labels"]
        
        # Create sort key: [label, -score, x1, y1, x2, y2]
        key = torch.stack([
            labels.to(torch.int64),
            (-scores).to(torch.float32),
            boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        ], dim=1)

        # Lexicographic sort: sort by last column first, then work backwards
        idx = torch.arange(key.size(0), device=key.device)
        for col in range(key.size(1) - 1, -1, -1):  # last -> first
            vals = key[idx, col]  # Use current permutation
            order = torch.argsort(vals, stable=True)
            idx = idx[order]

        return {
            "boxes": boxes[idx],
            "scores": scores[idx], 
            "labels": labels[idx],
        }

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
            "use_gpu_preprocess": self._use_gpu_preprocess,
        }
        return info

    @torch.inference_mode()
    def predict_batch(
            self, images: List[Union[Image.Image, np.ndarray]]
    ) -> List[List[dict]]:
        """
        Batch prediction for multiple images with optional GPU preprocessing.
        
        Parameters
        ----------
        images : List[Union[Image.Image, np.ndarray]]
            List of images to process in a single batch
            
        Returns
        -------
        List[List[dict]]
            List of prediction lists, one per input image. Each prediction dict contains:
            "label", "confidence", "l", "t", "r", "b"
        """
        if not images:
            return []

        # Convert all images to RGB PIL format for consistency
        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            elif isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img).convert("RGB"))
            else:
                raise TypeError("Not supported input image format")

        timer = self._create_timer()

        # Get target sizes for all images
        target_sizes = torch.tensor([img.size[::-1] for img in pil_images])

        if self._use_gpu_preprocess and self._gpu_preprocessor is not None:
            # GPU preprocessing path
            with timer.time_section('preprocess'):
                preprocessed = self._gpu_preprocessor.preprocess_batch(pil_images)
                pixel_values = preprocessed['pixel_values']
                if pixel_values.device != self._device:
                    pixel_values = pixel_values.to(self._device, non_blocking=True)
                pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)
        else:
            # CPU preprocessing path (original HF)
            with timer.time_section('preprocess'):
                inputs = self._image_preprocessor(images=pil_images, return_tensors="pt")
                pixel_values = inputs.pixel_values
                if self._device.type == "cuda":
                    pixel_values = pixel_values.to(self._device, non_blocking=True)
                    pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)
                else:
                    pixel_values = pixel_values.to(self._device)

        with timer.time_section('predict'):
            # Always use channels_last for better performance
            pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)
            
            # Cast to FP16 when using TensorRT
            if self._use_torch_compile_trt:
                pixel_values = pixel_values.to(dtype=self._trt_dtype)
            
            outputs = self._model(pixel_values=pixel_values)

        with timer.time_section('postprocess'):
            # Apply threshold hysteresis in compatibility mode
            threshold = self._threshold
            if self._compat_mode:
                threshold = max(0.0, threshold - 1e-4)
            
            results_list: List[Dict[str, Tensor]] = (
                self._image_postprocessor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=threshold,
                )
            )
            
            # Apply stable sorting in compatibility mode
            if self._compat_mode:
                results_list = [self._stable_sort_result(r) for r in results_list]
            
            # Move tensors to CPU in bulk to avoid hidden syncs later
            results_cpu = []
            for r in results_list:
                results_cpu.append({
                    "boxes": r["boxes"].detach().to("cpu"),
                    "scores": r["scores"].detach().to("cpu"),
                    "labels": r["labels"].detach().to("cpu"),
                })

        timer.finalize()
        self._store_timings(timer)

        # Convert results to standard format for each image
        all_predictions = []

        for img, results in zip(pil_images, results_cpu):
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

    @torch.inference_mode()
    def predict(self, orig_img: Union[Image.Image, np.ndarray]) -> Iterable[dict]:
        """
        Predict bounding boxes for a given image (single image version for compatibility).
        """
        # Use batch prediction internally
        results = self.predict_batch([orig_img])
        if results:
            for pred in results[0]:
                yield pred
