# GPU-accelerated preprocessing for RT-DETR
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
import torchvision.io
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from PIL import Image
import os
import contextlib
from pathlib import Path


def enable_strict_determinism():
    """
    Enable strict determinism for reproducible results.
    Disables TF32 and enables deterministic CUDA algorithms.
    
    IMPORTANT: Call this BEFORE any CUDA context is created (i.e., before the 
    first tensor hits GPU). Recommended usage in CI/compatibility mode:
    
    ```python
    if os.getenv("DOCLING_GPU_COMPAT_MODE", "").lower() in ("1", "true", "yes"):
        from fork.layout.gpu_preprocess import enable_strict_determinism
        enable_strict_determinism()
    ```
    
    Call once at process start for CI/compatibility runs.
    """
    # Disable TF32 (Ampere+ GPUs use this by default and it changes scores slightly)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # Enable deterministic kernels; disable autotuner
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    
    # Set workspace config for deterministic algorithms
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class GPUPreprocessor(nn.Module):
    """
    GPU-accelerated image preprocessing for RT-DETR.
    Maintains parity with HuggingFace RTDetrImageProcessor.
    """
    
    def __init__(
        self,
        size: Dict[str, int],
        do_pad: bool = False,
        pad_size: Optional[Dict[str, int]] = None,
        do_rescale: bool = True,
        rescale_factor: float = 1/255.0,
        do_normalize: bool = False,  # RT-DETR typically doesn't normalize
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.size = size
        self.do_pad = do_pad
        self.pad_size = pad_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.device = torch.device(device)  # Ensure proper device object
        self.dtype = dtype
        
        # Enable compatibility mode if environment variable is set
        self._compat_mode = os.getenv("DOCLING_GPU_COMPAT_MODE", "").lower() in ("1", "true", "yes")
        
        # Pre-compute mean/std tensors (only used if do_normalize=True)
        self.register_buffer('mean', torch.tensor(mean, dtype=dtype, device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std, dtype=dtype, device=device).view(1, 3, 1, 1))
        
        # Create resize transform based on size dict
        self.resize_op = self._make_resize(size)
        
        # Pre-allocated pinned staging tensor for large batches (avoid allocator churn)
        self._staging_cache = {}  # key: (B_bucket, H, W, C, dtype) -> pinned tensor
        
    def _make_resize(self, size_dict: Dict[str, int]):
        """Create appropriate resize operation based on size specification."""
        if "height" in size_dict and "width" in size_dict:
            # Exact size resize (no aspect ratio preservation)
            # Always use antialias=False to match PIL bilinear behavior
            # Wrap in contiguity-ensuring function for robustness
            target_h, target_w = size_dict["height"], size_dict["width"]
            
            def resize_with_contiguity(x):
                # Ensure input is contiguous before resize
                x_cont = x.contiguous()
                resized = F.resize(
                    x_cont,
                    [target_h, target_w],
                    interpolation=T.InterpolationMode.BILINEAR,
                    antialias=False
                )
                return resized.contiguous()  # Ensure output is contiguous too
            
            return resize_with_contiguity
        elif "shortest_edge" in size_dict and "longest_edge" in size_dict:
            # Custom resize respecting shortest/longest edge constraints
            se = size_dict["shortest_edge"]
            le = size_dict["longest_edge"]
            
            class ShortLongResize(nn.Module):
                def forward(self, x):
                    # x shape: (B, C, H, W)
                    B, C, H, W = x.shape
                    
                    # Determine scale based on shortest edge
                    short_edge = min(H, W)
                    scale = se / short_edge
                    
                    # Calculate new dimensions
                    new_h = int(round(H * scale))
                    new_w = int(round(W * scale))
                    
                    # Clamp by longest edge if needed
                    longest_new = max(new_h, new_w)
                    if longest_new > le:
                        scale2 = le / longest_new
                        new_h = int(round(new_h * scale2))
                        new_w = int(round(new_w * scale2))
                    
                    # Ensure contiguity for consistent performance
                    x_cont = x.contiguous()
                    resized = F.resize(
                        x_cont, 
                        [new_h, new_w], 
                        interpolation=T.InterpolationMode.BILINEAR, 
                        antialias=False
                    )
                    return resized.contiguous()
            
            return ShortLongResize()
        else:
            raise ValueError(f"Unsupported size specification: {size_dict}")
    
    def _get_staging_tensor(self, shape, dtype):
        """Get pinned staging tensor with bucket-based allocation to avoid churn."""
        B, H, W, C = shape
        # Round batch size up to next power of 2 or common size
        B_bucket = max(8, 1 << (B-1).bit_length()) if B > 0 else 8  # 8, 16, 32, 64...
        bucket_key = (B_bucket, H, W, C, dtype)
        
        if bucket_key not in self._staging_cache:
            self._staging_cache[bucket_key] = torch.empty(
                (B_bucket, H, W, C), dtype=dtype, pin_memory=True
            )
        return self._staging_cache[bucket_key][:B]  # View into larger buffer
    
    @torch.no_grad()
    def preprocess_batch(
        self, 
        images: List[Union[Image.Image, np.ndarray, torch.Tensor, str, Path]]
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess a batch of images on GPU.
        
        Args:
            images: List of PIL Images, numpy arrays, torch tensors, or file paths
            
        Returns:
            Dictionary with 'pixel_values' and optionally 'pixel_mask'
        """
        if not images:
            return {'pixel_values': torch.empty(0, 3, 0, 0, device=self.device, dtype=self.dtype)}
        
        # Convert all images to torch tensors on CPU first
        cpu_tensors = []
        for img in images:
            if isinstance(img, (str, Path)):
                # Direct JPEG/PNG decode using libjpeg-turbo via torchvision.io
                img_path = str(img)
                try:
                    img_tensor = torchvision.io.read_image(img_path)  # Already CHW uint8
                    if img_tensor.shape[0] == 1:  # Grayscale
                        img_tensor = img_tensor.repeat(3, 1, 1)
                except Exception:
                    # Fallback to PIL if torchvision.io fails
                    with Image.open(img_path) as pil_img:
                        img_np = np.array(pil_img.convert('RGB'))
                        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
            elif isinstance(img, Image.Image):
                # PIL Image -> numpy -> torch
                img_np = np.array(img.convert('RGB'))
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # HWC -> CHW
            elif isinstance(img, np.ndarray):
                # Numpy array -> torch
                if img.ndim == 2:  # Grayscale
                    img = np.stack([img] * 3, axis=-1)
                elif img.ndim == 3 and img.shape[2] == 1:  # Single channel
                    img = np.repeat(img, 3, axis=2)
                img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
            elif isinstance(img, torch.Tensor):
                if img.dim() == 2:  # Grayscale
                    img_tensor = img.unsqueeze(0).repeat(3, 1, 1)
                else:
                    img_tensor = img
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
            
            cpu_tensors.append(img_tensor)
        
        # Stack into batch and move to GPU
        batch_cpu = torch.stack(cpu_tensors)
        is_u8 = (batch_cpu.dtype == torch.uint8)
        
        # Use pre-allocated pinned staging tensor for faster transfer
        if is_u8 and torch.cuda.is_available() and self.device.type == "cuda":
            # Get staging tensor from bucket cache (BCHW -> BHWC for staging)
            batch_shape_hwc = (batch_cpu.shape[0], batch_cpu.shape[2], batch_cpu.shape[3], batch_cpu.shape[1])
            staging_tensor = self._get_staging_tensor(batch_shape_hwc, batch_cpu.dtype)
            
            # Copy to staging tensor (CHW -> HWC conversion)
            batch_hwc = batch_cpu.permute(0, 2, 3, 1)
            staging_tensor.copy_(batch_hwc)
            batch_gpu = staging_tensor.permute(0, 3, 1, 2).to(self.device, non_blocking=True)
        else:
            batch_gpu = batch_cpu.to(self.device, non_blocking=True)
        
        # Convert to target dtype
        batch_gpu = batch_gpu.to(self.dtype)
        
        # Rescale if needed (simpler, no sync)
        if self.do_rescale and is_u8:
            batch_gpu.mul_(self.rescale_factor)
        
        # Apply resize
        batch_resized = self.resize_op(batch_gpu)
        
        # Normalize only if do_normalize=True
        if self.do_normalize:
            batch_normalized = (batch_resized - self.mean) / self.std
        else:
            batch_normalized = batch_resized
        
        # Handle padding if needed
        if self.do_pad:
            # Assert fixed size for predictable padding behavior
            assert "height" in self.size and "width" in self.size, \
                   "V1 requires fixed-size resize when pad_size=None"
            batch_padded, pixel_mask = self._apply_padding(batch_normalized)
            return {
                'pixel_values': batch_padded,
                'pixel_mask': pixel_mask
            }
        else:
            return {
                'pixel_values': batch_normalized
            }
    
    def _apply_padding(
        self, 
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply padding to batch and create pixel masks.
        
        Args:
            batch: Tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of (padded_batch, pixel_mask)
        """
        B, C, H, W = batch.shape
        
        if self.pad_size is not None:
            # Pad to specified size
            target_h = self.pad_size["height"]
            target_w = self.pad_size["width"]
        else:
            # Pad to max size in batch
            target_h = H  # All images same size after resize
            target_w = W
        
        if target_h == H and target_w == W:
            # No padding needed
            pixel_mask = torch.ones((B, H, W), dtype=torch.int64, device=self.device)
            return batch, pixel_mask
        
        # Use F.pad for single-kernel efficiency: (left, right, top, bottom)
        pad_right = target_w - W
        pad_bottom = target_h - H
        padded = F.pad(batch, (0, pad_right, 0, pad_bottom), mode='constant', value=0.0)
        
        # Create pixel mask efficiently using F.pad
        mask_template = torch.ones((B, H, W), dtype=torch.int64, device=self.device)
        pixel_mask = F.pad(mask_template, (0, pad_right, 0, pad_bottom), mode='constant', value=0)
        
        return padded, pixel_mask


class GPUPreprocessorV2(nn.Module):
    """
    Optimized V2: Uses channels_last memory format for better performance.
    """
    
    def __init__(
        self,
        size: Dict[str, int],
        do_pad: bool = False,
        pad_size: Optional[Dict[str, int]] = None,
        do_rescale: bool = True,
        rescale_factor: float = 1/255.0,
        do_normalize: bool = False,  # RT-DETR typically doesn't normalize
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float32,
        return_channels_last: bool = False,  # Set True if model uses channels_last
    ):
        super().__init__()
        self.size = size
        self.do_pad = do_pad
        self.pad_size = pad_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.device = torch.device(device)  # Ensure proper device object
        self.dtype = dtype
        self.return_channels_last = return_channels_last
        
        # Enable compatibility mode if environment variable is set
        self._compat_mode = os.getenv("DOCLING_GPU_COMPAT_MODE", "").lower() in ("1", "true", "yes")
        
        # Pre-compute mean/std for NHWC format (only used if do_normalize=True)
        self.register_buffer('mean', torch.tensor(mean, dtype=dtype, device=device).view(1, 1, 1, 3))
        self.register_buffer('std', torch.tensor(std, dtype=dtype, device=device).view(1, 1, 1, 3))
        
        # Create resize operation
        self.resize_op = self._make_resize(size)
        
        # Pre-allocated pinned staging tensor for large batches (avoid allocator churn)
        self._staging_cache = {}  # key: (B_bucket, H, W, C, dtype) -> pinned tensor
        
        # Persistent CUDA streams and events (avoid creation overhead per call)
        if self.device.type == "cuda":
            self._stream_h2d = torch.cuda.Stream()
            self._stream_compute = torch.cuda.Stream()
            self._h2d_event = torch.cuda.Event()
            
            # Warmup: initialize CUDA context and JIT compile kernels to avoid first-call overhead
            with torch.cuda.stream(self._stream_compute):
                dummy = torch.ones(1, 224, 224, 3, device=self.device, dtype=self.dtype)
                dummy.mul_(1.0)
                dummy_nchw = dummy.permute(0, 3, 1, 2).contiguous()
                dummy_resized = F.resize(dummy_nchw, [224, 224], 
                                       interpolation=T.InterpolationMode.BILINEAR, antialias=False)
                dummy_result = dummy_resized.contiguous().permute(0, 2, 3, 1).contiguous()
            torch.cuda.synchronize()  # Ensure warmup completes
        else:
            self._stream_h2d = None
            self._stream_compute = None
            self._h2d_event = None
    
    def _make_resize(self, size_dict: Dict[str, int]):
        """Create appropriate resize operation based on size specification."""
        # V2 only supports fixed size for optimal performance
        assert "height" in size_dict and "width" in size_dict, "V2 supports fixed size only"
        
        target_h = size_dict["height"]
        target_w = size_dict["width"]
        
        def resize_exact(x):
            # x is NHWC - force contiguity to avoid F.resize slow path
            x_nchw = x.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW, ensure contiguous
            resized = F.resize(
                x_nchw,
                [target_h, target_w],
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=False
            )
            return resized.contiguous().permute(0, 2, 3, 1).contiguous()  # Back to NHWC, force contiguous
        
        return resize_exact
    
    def _get_staging_tensor(self, shape, dtype):
        """Get pinned staging tensor with bucket-based allocation to avoid churn."""
        B, H, W, C = shape
        # Round batch size up to next power of 2 or common size
        B_bucket = max(8, 1 << (B-1).bit_length()) if B > 0 else 8  # 8, 16, 32, 64...
        bucket_key = (B_bucket, H, W, C, dtype)
        
        if bucket_key not in self._staging_cache:
            self._staging_cache[bucket_key] = torch.empty(
                (B_bucket, H, W, C), dtype=dtype, pin_memory=True
            )
        return self._staging_cache[bucket_key][:B]  # View into larger buffer
    
    @torch.no_grad()
    def preprocess_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]]
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess batch with optimized memory layout.
        """
        if not images:
            return {'pixel_values': torch.empty(0, 3, 0, 0, device=self.device, dtype=self.dtype)}
        
        # Convert to numpy arrays in HWC format
        np_images = []
        for img in images:
            if isinstance(img, Image.Image):
                np_images.append(np.array(img.convert('RGB'), dtype=np.uint8))
            elif isinstance(img, np.ndarray):
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                np_images.append(img.astype(np.uint8))
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
        
        # Stack into batch (NHWC format)
        batch_np = np.stack(np_images)
        
        # Use persistent streams (created in __init__)
        stream_h2d = self._stream_h2d
        stream_compute = self._stream_compute
        h2d_event = self._h2d_event
        
        # Create pinned tensor using staging buffer to avoid allocator churn
        batch_tensor = torch.from_numpy(batch_np)
        if self.device.type == "cuda":
            # Get staging tensor from bucket cache (already NHWC)
            staging_tensor = self._get_staging_tensor(batch_tensor.shape, batch_tensor.dtype)
            
            # Copy to staging tensor
            staging_tensor.copy_(batch_tensor)
            batch_pinned = staging_tensor
        else:
            batch_pinned = batch_tensor
        
        if stream_h2d:
            with torch.cuda.stream(stream_h2d):
                batch_gpu = batch_pinned.to(self.device, non_blocking=True)
                # Keep in NHWC format (already contiguous from numpy)
                # Record event when H2D transfer completes
                if h2d_event:
                    h2d_event.record()
        else:
            batch_gpu = batch_pinned.to(self.device)
        
        # Make compute stream wait on H2D completion using event (more efficient than wait_stream)
        if stream_compute and h2d_event:
            stream_compute.wait_event(h2d_event)
        
        with (torch.cuda.stream(stream_compute) if stream_compute else contextlib.nullcontext()):
            # Convert to float
            batch_float = batch_gpu.to(self.dtype)
            # Rescale if needed (in-place for efficiency)
            if self.do_rescale:
                batch_float.mul_(self.rescale_factor)
            
            # Resize (internally converts to NCHW and back)
            batch_resized = self.resize_op(batch_float)
            
            # Normalize only if do_normalize=True
            if self.do_normalize:
                batch_normalized = (batch_resized - self.mean) / self.std
            else:
                batch_normalized = batch_resized
            
            # Convert to final format based on model preference
            if self.return_channels_last:
                # Keep NHWC format with channels_last memory layout for optimal model performance
                batch_final = batch_normalized.contiguous(memory_format=torch.channels_last)
            else:
                # Convert to NCHW for standard model input
                batch_final = batch_normalized.permute(0, 3, 1, 2).contiguous()
        
        # Handle padding
        if self.do_pad:
            batch_padded, pixel_mask = self._apply_padding(batch_final)
            result = {
                'pixel_values': batch_padded,
                'pixel_mask': pixel_mask
            }
        else:
            result = {
                'pixel_values': batch_final
            }
        
        # Ensure compute is done before returning (optional sync)
        if stream_compute:
            torch.cuda.current_stream().wait_stream(stream_compute)
        
        return result
    
    def _apply_padding(
        self,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply padding to batch (NCHW format)."""
        B, C, H, W = batch.shape
        
        if self.pad_size is not None:
            target_h = self.pad_size["height"]
            target_w = self.pad_size["width"]
        else:
            target_h = H
            target_w = W
        
        if target_h == H and target_w == W:
            pixel_mask = torch.ones((B, H, W), dtype=torch.int64, device=self.device)
            return batch, pixel_mask
        
        # Use F.pad for single-kernel efficiency: (left, right, top, bottom)
        pad_right = target_w - W
        pad_bottom = target_h - H
        padded = F.pad(batch, (0, pad_right, 0, pad_bottom), mode='constant', value=0.0)
        
        # Create pixel mask efficiently using F.pad
        mask_template = torch.ones((B, H, W), dtype=torch.int64, device=self.device)
        pixel_mask = F.pad(mask_template, (0, pad_right, 0, pad_bottom), mode='constant', value=0)
        
        return padded, pixel_mask