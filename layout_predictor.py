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
from typing import Dict, List, Set, Union, Optional

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import Tensor
from transformers import AutoModelForObjectDetection, RTDetrImageProcessor

from docling_ibm_models.layoutmodel.labels import LayoutLabels
from optimization_utils import enable_fast_backends, safe_autocast, prepare_model_for_infer

_log = logging.getLogger(__name__)

# Global lock for model initialization to prevent threading issues
_model_init_lock = threading.Lock()


class CudaTimer:
    """Accurate GPU timing with CUDA events"""
    def __init__(self, enabled=True):
        self.enabled = enabled and torch.cuda.is_available()
        if self.enabled:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        self.ms = 0.0
    
    def __enter__(self):
        if self.enabled:
            self.start_event.record()
        else:
            self.t0 = time.perf_counter()
        return self
    
    def __exit__(self, *_):
        if self.enabled:
            self.end_event.record()
            self.end_event.synchronize()
            self.ms = self.start_event.elapsed_time(self.end_event)
        else:
            self.ms = (time.perf_counter() - self.t0) * 1e3


class StageTimes:
    """Accumulate and report stage timings"""
    def __init__(self):
        self.data = {}
    
    def add(self, name, ms):
        self.data[name] = self.data.get(name, 0) + ms
    
    def report(self, total_ms=None):
        tot = total_ms if total_ms is not None else sum(self.data.values())
        parts = " | ".join(f"{k}:{v:.1f}ms({100*v/tot:.0f}%)" for k, v in self.data.items())
        return f"{tot:.1f}ms total -> {parts}"


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
            enable_timing: bool = False,
            enable_nms: bool = False,
            max_batch_size: int = 128,
    ):
        """
        Provide the artifact path that contains the LayoutModel file

        Parameters
        ----------
        artifact_path: Path for the model torch file.
        device: (Optional) device to run the inference.
        num_threads: (Optional) Number of threads to run the inference if device = 'cpu'
        max_batch_size: (Optional) Maximum batch size for processing images. Default 128.

        Raises
        ------
        FileNotFoundError when the model's torch file is missing
        """
        # Blacklisted classes
        self._black_classes = blacklist_classes  # set(["Form", "Key-Value Region"])
        self._enable_timing = enable_timing
        self._enable_nms = enable_nms
        self._max_batch_size = max_batch_size

        # Canonical classes
        self._labels = LayoutLabels()

        # Set basic params
        self._threshold = base_threshold  # Score threshold

        # Set number of threads for CPU
        self._device = torch.device(device)
        self._num_threads = num_threads
        if device == "cpu":
            torch.set_num_threads(self._num_threads)
        
        # Explicitly enable optimizations (not on import)
        enable_fast_backends()
        if self._device.type == "cuda":
            torch.set_float32_matmul_precision("high")

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
                artifact_path, config=self._model_config
            )
            # Use optimization utils for proper model preparation
            self._model = prepare_model_for_infer(self._model, self._device)
            
            # Warmup to stabilize cuDNN autotune and avoid first-run penalty
            if self._device.type == "cuda":
                dummy = torch.zeros(1, 3, 640, 640, device=self._device).contiguous(memory_format=torch.channels_last)
                with safe_autocast(self._device, dtype=torch.float16), torch.inference_mode():
                    _ = self._model(pixel_values=dummy)
                torch.cuda.synchronize()
                _log.debug("Model warmup complete")

        # Set classes map
        self._model_name = type(self._model).__name__
        if self._model_name == "RTDetrForObjectDetection":
            self._classes_map = self._labels.shifted_canonical_categories()
            self._label_offset = 1
        else:
            self._classes_map = self._labels.canonical_categories()
            self._label_offset = 0
        
        # Pre-compute name to ID mapping for blacklist filtering
        self._name_to_id = {name: i for i, name in enumerate(self._classes_map)}
        
        # Precompute blacklist tensor once
        if self._black_classes:
            blk_ids = [self._name_to_id[n] - self._label_offset for n in self._black_classes]
            self._blk_ids = torch.tensor(blk_ids, device=self._device, dtype=torch.long)
        else:
            self._blk_ids = None

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
    def predict_batch(
            self, images: List[Union[Image.Image, np.ndarray]]
    ) -> List[List[dict]]:
        """
        Batch prediction for multiple images - more efficient than calling predict() multiple times.
        Processes images in chunks of max_batch_size to optimize GPU memory usage.

        Returns a list of per-image predictions.
        """
        if not images:
            return []

        # Convert all images to RGB PIL format first
        pil_images = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            elif isinstance(img, np.ndarray):
                pil_images.append(Image.fromarray(img).convert("RGB"))
            else:
                raise TypeError("Not supported input image format")
        
        # Process in chunks if needed
        if len(pil_images) > self._max_batch_size:
            all_predictions = []
            for i in range(0, len(pil_images), self._max_batch_size):
                chunk = pil_images[i:i + self._max_batch_size]
                chunk_preds = self._predict_batch_chunk(chunk)
                all_predictions.extend(chunk_preds)
            
            if self._enable_timing:
                _log.info(f"[layout.batch] Processed {len(pil_images)} images in {len(range(0, len(pil_images), self._max_batch_size))} chunks of size {self._max_batch_size}")
            
            return all_predictions
        else:
            # Process all at once if within batch size limit
            return self._predict_batch_chunk(pil_images)
    
    def _predict_batch_chunk(
            self, pil_images: List[Image.Image]
    ) -> List[List[dict]]:
        """
        Process a single chunk of images (internal method).
        """
        # Performance timing (only if enabled to avoid synchronization overhead)
        if self._enable_timing:
            st = StageTimes()
            wall_t0 = time.perf_counter()

        # Target sizes for postprocess (create on same device as model)
        target_sizes = torch.tensor([img.size[::-1] for img in pil_images], device=self._device)

        # GPU-accelerated preprocessing (resize + normalize on GPU)
        if self._enable_timing:
            with CudaTimer() as t_pre:
                pixel_values = rtdetr_preprocess_tensor(pil_images, device=self._device)
            st.add("pre", t_pre.ms)
        else:
            pixel_values = rtdetr_preprocess_tensor(pil_images, device=self._device)
        
        # Forward pass with safe autocast (handles cuda/mps/cpu)
        if self._enable_timing:
            with CudaTimer() as t_fwd, safe_autocast(self._device, dtype=torch.float16):
                outputs = self._model(pixel_values=pixel_values)
            st.add("fwd", t_fwd.ms)
        else:
            with safe_autocast(self._device, dtype=torch.float16):
                outputs = self._model(pixel_values=pixel_values)

        # Post-process results
        def post_process():
            boxes, scores, labels, batch_idx, splits = self.post_process_object_detection_fast(
                outputs, threshold=self._threshold, target_sizes=target_sizes, use_focal_loss=True
            )

            # Early GPU culling of small/extreme aspect ratio boxes (optional)
            if self._device.type == "cuda" and boxes.numel() > 0:
                w = (boxes[:, 2] - boxes[:, 0]).clamp_min_(0)
                h = (boxes[:, 3] - boxes[:, 1]).clamp_min_(0)
                wh = w * h
                ar = (w.clamp_min_(1e-6)) / (h.clamp_min_(1e-6))
                keep = (wh >= 9.0) & (ar <= 20) & (ar >= 1/20)
                # No need for keep.any() check - indexing with empty mask is fine
                boxes, scores, labels, batch_idx = boxes[keep], scores[keep], labels[keep], batch_idx[keep]

            # GPU-accelerated blacklist filtering
            if self._blk_ids is not None and labels.numel() > 0:
                keep = ~torch.isin(labels, self._blk_ids)
                boxes, scores, labels, batch_idx = boxes[keep], scores[keep], labels[keep], batch_idx[keep]

            # Batched NMS (only if enabled - can add overhead)
            if self._enable_nms and self._device.type == "cuda" and boxes.numel() > 0:
                # Avoid .item() sync - keep label_stride on GPU
                label_stride = (labels.max() + 1) if labels.numel() > 0 else labels.new_tensor(1)
                kept = torchvision.ops.batched_nms(
                    boxes, scores, labels + batch_idx * label_stride, iou_threshold=0.5
                )
                boxes, scores, labels, batch_idx = boxes[kept], scores[kept], labels[kept], batch_idx[kept]
            
            return boxes, scores, labels, batch_idx, splits
        
        if self._enable_timing:
            with CudaTimer() as t_post:
                boxes, scores, labels, batch_idx, splits = post_process()
            st.add("post", t_post.ms)
        else:
            boxes, scores, labels, batch_idx, splits = post_process()
        
        # Detection count logging for spotting outliers
        if self._enable_timing and scores.numel() > 0:
            det_total = int(scores.numel())
            det_per_page = torch.bincount(batch_idx, minlength=len(pil_images)).detach().cpu().tolist()
            _log.debug(f"[layout.det] total={det_total} | per_page={det_per_page[:8]}{'...' if len(det_per_page)>8 else ''}")
        
        # Use splits for efficient per-image slicing (no CPU boolean masks)
        boxes, scores, labels, batch_idx, splits = post_process()
        boxes_cpu = boxes.detach().cpu()
        scores_cpu = scores.detach().cpu() 
        labels_cpu = labels.detach().cpu()

        all_predictions = []
        for i, (lo, hi) in enumerate(splits):
            b_i = boxes_cpu[lo:hi]
            s_i = scores_cpu[lo:hi]
            l_i = labels_cpu[lo:hi]
            if hi > lo:
                coords = b_i.numpy().tolist()
                confs  = s_i.numpy().tolist()
                labs   = l_i.numpy().tolist()
                w, h = pil_images[i].size
                preds = [
                    {
                        "l": float(c[0]), "t": float(c[1]),
                        "r": float(c[2]), "b": float(c[3]),
                        "label": self._classes_map[int(lbl) + self._label_offset],
                        "confidence": float(sc),
                    }
                    for c, sc, lbl in zip(coords, confs, labs)
                ]
            else:
                preds = []
            all_predictions.append(preds)

        # Report timing (only if enabled)
        if self._enable_timing:
            wall_ms = (time.perf_counter() - wall_t0) * 1e3
            total_ms = sum(st.data.values())
            pps = len(pil_images) / (total_ms / 1e3) if total_ms > 0 else 0
            _log.info(f"[layout.gpu] {st.report(total_ms)} | pages={len(pil_images)} | {pps:.1f} pages/s")
            _log.info(f"[layout.wall] {wall_ms:.1f}ms for {len(pil_images)} pages ({len(pil_images)/(wall_ms/1e3):.1f} pages/s)")
            
            if torch.cuda.is_available():
                peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
                _log.debug(f"[gpu.mem] peak_allocated={peak_gb:.2f} GB")

        return all_predictions
        # results_list: List[Dict[str, Tensor]] = (
        #     self._image_processor.post_process_object_detection(
        #         outputs,
        #         target_sizes=target_sizes,
        #         threshold=self._threshold,
        #     )
        # )
        # t_post = time.perf_counter()
        #
        # # Simple timing output
        # print(
        #     f"Layout: preprocess={t_pre-t0:.2f}s, forward={t_fwd-t_pre:.2f}s, post={t_post-t_fwd:.2f}s"
        # )
        #
        # # Convert results to standard format for each image
        # all_predictions: List[List[dict]] = []
        # for img, results in zip(pil_images, results_list):
        #     w, h = img.size
        #     predictions = []
        #     for score, label_id, box in zip(
        #             results["scores"], results["labels"], results["boxes"]
        #     ):
        #         score = float(score.item())
        #         label_id = int(label_id.item()) + self._label_offset
        #         label_str = self._classes_map[label_id]
        #
        #         # Filter out blacklisted classes
        #         if label_str in self._black_classes:
        #             continue
        #
        #         bbox_float = [float(b.item()) for b in box]
        #         l = min(w, max(0, bbox_float[0]))
        #         t = min(h, max(0, bbox_float[1]))
        #         r = min(w, max(0, bbox_float[2]))
        #         b = min(h, max(0, bbox_float[3]))
        #
        #         predictions.append(
        #             {
        #                 "l": l,
        #                 "t": t,
        #                 "r": r,
        #                 "b": b,
        #                 "label": label_str,
        #                 "confidence": score,
        #             }
        #         )
        #     all_predictions.append(predictions)
        #
        # return all_predictions

    @torch.inference_mode()
    def post_process_object_detection_fast(
            self,
            outputs,
            threshold: float = 0.5,
            target_sizes: Optional[torch.Tensor] = None,
            use_focal_loss: bool = True,
    ):
        """
        Vectorized, GPU-friendly variant of RTDetrImageProcessor.post_process_object_detection.

        Returns:
          boxes:     [N,4] (l,t,r,b)
          scores:    [N]
          labels:    [N]   (int64)
          batch_idx: [N]   (int64) index of original image in batch
          splits:    Optional[List[Tuple[start,end]]] for per-image slicing (device-free metadata)
        """
        # logits: [B, Q, C]  pred_boxes: [B, Q, 4]  in cxcywh relative coords
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes.to(torch.float32)  # Ensure float32 for geometry
        device = logits.device
        B, Q, C = logits.shape

        # Convert to xyxy in absolute pixels if target_sizes provided
        # cxcywh -> xyxy
        cx, cy, w, h = pred_boxes.unbind(-1)
        x0 = cx - 0.5 * w
        y0 = cy - 0.5 * h
        x1 = cx + 0.5 * w
        y1 = cy + 0.5 * h
        boxes_xyxy = torch.stack([x0, y0, x1, y1], dim=-1)

        if target_sizes is not None:
            if isinstance(target_sizes, list):
                target_sizes = torch.as_tensor(target_sizes, device=device)
            elif target_sizes.device != device:
                target_sizes = target_sizes.to(device)
            # target_sizes: [B,2]= (H,W)
            img_h, img_w = target_sizes.unbind(1)  # [B], [B]
            scale = torch.stack([img_w, img_h, img_w, img_h], dim=1).unsqueeze(1)  # [B,1,4]
            boxes_xyxy = boxes_xyxy * scale

        # Scores/labels selection
        if use_focal_loss:
            # sigmoid over classes, then take top-Q over flattened (Q*C)
            prob = logits.sigmoid()  # [B,Q,C]
            prob_flat = prob.flatten(1)  # [B,Q*C]
            topk_scores, topk_idx = torch.topk(prob_flat, k=Q, dim=1)  # [B,Q]
            labels = topk_idx % C  # [B,Q]
            query_idx = topk_idx // C  # [B,Q]
            # gather boxes by chosen queries
            gather_idx = query_idx.unsqueeze(-1).expand(B, Q, 4)  # [B,Q,4]
            boxes = boxes_xyxy.gather(1, gather_idx)  # [B,Q,4]
            scores = topk_scores  # [B,Q]
        else:
            # Softmax over classes (excluding background last class)
            prob = torch.softmax(logits[..., :-1], dim=-1)  # [B,Q,C-1]
            scores, labels = prob.max(dim=-1)  # [B,Q]
            # Optional: topQ again to cap per-image detections to Q
            topk_scores, topk_idx = torch.topk(scores, k=Q, dim=1)
            scores = topk_scores
            labels = labels.gather(1, topk_idx)
            gather_idx = topk_idx.unsqueeze(-1).expand(B, Q, 4)
            boxes = boxes_xyxy.gather(1, gather_idx)

        # Threshold mask (vectorized)
        keep_mask = scores > threshold  # [B,Q]
        # Flatten batch
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(scores)  # [B,Q]

        boxes = boxes[keep_mask]  # [N,4]
        scores = scores[keep_mask]  # [N]
        labels = labels[keep_mask].to(torch.int64)
        batch_idx = batch_idx[keep_mask].to(torch.int64)

        # Optional: per-image splits metadata (CPU small list, OK)
        if B > 1:
            # Compute counts per image without sync to CPU tensors
            counts = keep_mask.sum(dim=1)  # [B]
            # prefix sums on CPU for easy slicing later
            counts_cpu = counts.detach().cpu().tolist()
            offsets = [0]
            for c in counts_cpu:
                offsets.append(offsets[-1] + int(c))
            splits = [(offsets[i], offsets[i + 1]) for i in range(B)]
        else:
            splits = [(0, boxes.shape[0])]

        return boxes, scores, labels, batch_idx, splits


# def rtdetr_preprocess_tensor(pil_list, device="cuda"):
#     # Convert PIL -> RGB np.uint8
#     rgb_list = [np.array(im.convert("RGB"), dtype=np.uint8) for im in pil_list]
#     t = torch.stack([torch.from_numpy(x) for x in rgb_list])          # [B,H,W,3], uint8
#     t = t.permute(0, 3, 1, 2).contiguous().to(torch.float32)          # [B,3,H,W], float32
#     t = t.div_(255.0)                                                 # rescale to [0,1]
#     t = F.interpolate(t, size=(640, 640), mode="bilinear", align_corners=False)  # warp (no AR)
#     t = t.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
#     return t  # feed as pixel_values


def rtdetr_preprocess_tensor(pil_list, device):
    import numpy as np
    import torch.nn.functional as F

    # One pass → one big array [B,H,W,3] uint8
    rgb = np.stack([np.asarray(im.convert("RGB"), dtype=np.uint8) for im in pil_list], axis=0)
    t = torch.from_numpy(rgb)  # CPU uint8, shared memory, zero-copy

    if device.type == "cuda":
        t = t.pin_memory()

    # [B,H,W,3] -> [B,3,H,W] on CPU; then a single non_blocking H2D
    t = t.permute(0, 3, 1, 2).contiguous()
    t = t.to(device, non_blocking=True).to(torch.float32)
    t = t.div_(255.0)
    t = F.interpolate(t, size=(640, 640), mode="bilinear", align_corners=False)
    return t.contiguous(memory_format=torch.channels_last)
