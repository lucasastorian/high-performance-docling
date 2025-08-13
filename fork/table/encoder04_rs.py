#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

import logging
import os

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.utils.fusion import fuse_conv_bn_eval
import torchvision

import docling_ibm_models.tableformer.settings as s

LOG_LEVEL = logging.INFO
# LOG_LEVEL = logging.DEBUG


class Encoder04(nn.Module):
    """
    ResNet-18 stem producing 256-channel feature maps at stride 16.
    Output: NCHW [B,256,enc_image_size,enc_image_size]
    """

    def __init__(self, enc_image_size: int, enc_dim: int = 512):
        super().__init__()
        
        # Backend optimizations (Optimization 6)
        # Enable TF32 for faster matmuls on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")  # helps linear/proj in later stages
        # Note: cudnn.benchmark will be enabled during warmup, not here
        
        self.enc_image_size = enc_image_size
        self._encoder_dim = 256  # ResNet18 layer3 output channels

        # Keep the same module names/structure expected by the checkpoint:
        resnet = torchvision.models.resnet18(weights=None)
        trunk = list(resnet.children())[:-3]  # conv1..layer3
        self._resnet = nn.Sequential(*trunk)  # <-- name kept as _resnet

        self._adaptive_pool = nn.AdaptiveAvgPool2d((self.enc_image_size, self.enc_image_size))
        
        # CUDA Graphs support - ON by default
        self._use_graphs = bool(int(os.getenv("ENCODER_USE_GRAPHS", "1")))
        self._gr_bs = int(os.getenv("ENCODER_BLOCK_BS", "32"))
        self._gr_in = None
        self._gr_out = None
        self._gr = None
        self._gr_reuses_output = True  # caller must .clone() if they need a fresh tensor
        self._fused = False  # Track if Conv+BN fusion has been applied
        self._pool_checked = False  # Track if we've checked for no-op pool
        
        # BF16 support - OFF by default (controlled by TABLE_ENCODER_BF16)
        self._use_bf16 = bool(int(os.getenv("TABLE_ENCODER_BF16", "0")))
        if self._use_bf16:
            self._log().info("BF16 enabled for encoder via TABLE_ENCODER_BF16=1")

    def _log(self):
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)
    
    def _as_device(self, d):
        """Convert string or device to torch.device"""
        return d if isinstance(d, torch.device) else torch.device(d)

    def get_encoder_dim(self) -> int:
        return self._encoder_dim
    
    def _fuse_conv_bn(self):
        """Fuse Conv+BN layers for inference (Optimization 3)"""
        if self._fused:
            return
            
        self.eval()  # Must be in eval mode for fusion
        
        # Fuse conv1+bn1 in the stem
        modules = dict(self._resnet.named_children())
        if '0' in modules and '1' in modules:  # conv1 and bn1
            conv1 = modules['0']
            bn1 = modules['1']
            if isinstance(conv1, nn.Conv2d) and isinstance(bn1, nn.BatchNorm2d):
                fused_conv1 = fuse_conv_bn_eval(conv1, bn1)
                # Replace in the sequential
                new_modules = []
                for name, module in self._resnet.named_children():
                    if name == '0':
                        new_modules.append(fused_conv1)
                    elif name == '1':
                        new_modules.append(nn.Identity())  # Replace BN with identity
                    else:
                        new_modules.append(module)
                self._resnet = nn.Sequential(*new_modules)
        
        # Fuse Conv+BN in BasicBlocks (layers 1, 2, 3)
        for module in self._resnet.modules():
            if isinstance(module, torchvision.models.resnet.BasicBlock):
                # Fuse conv1+bn1
                if hasattr(module, 'conv1') and hasattr(module, 'bn1'):
                    module.conv1 = fuse_conv_bn_eval(module.conv1, module.bn1)
                    module.bn1 = nn.Identity()
                # Fuse conv2+bn2
                if hasattr(module, 'conv2') and hasattr(module, 'bn2'):
                    module.conv2 = fuse_conv_bn_eval(module.conv2, module.bn2)
                    module.bn2 = nn.Identity()
                # Handle downsample if present
                if module.downsample is not None:
                    # downsample is typically Sequential(conv, bn)
                    if len(module.downsample) == 2:
                        conv = module.downsample[0]
                        bn = module.downsample[1]
                        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
                            module.downsample = nn.Sequential(
                                fuse_conv_bn_eval(conv, bn),
                                nn.Identity()
                            )
        
        self._fused = True
        self._log().info("Conv+BN layers fused for inference")
        
        # Set module to use channels-last for activations (not weights!)
        self._resnet.to(memory_format=torch.channels_last)

    def _check_adaptive_pool(self, H=448, W=448, device="cuda"):
        """Check if AdaptiveAvgPool2d is a no-op and replace with Identity (Optimization 2)"""
        if self._pool_checked:
            return
        
        dev = self._as_device(device)
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, H, W, device=dev).to(memory_format=torch.channels_last)
            out = self._resnet(dummy)
        
        if out.shape[-2:] == (self.enc_image_size, self.enc_image_size):
            self._adaptive_pool = nn.Identity()
            self._log().info(f"AdaptiveAvgPool2d is no-op at size {self.enc_image_size}, replaced with Identity")
        
        self._pool_checked = True

    def _maybe_capture(self, C=3, H=448, W=448, device="cuda"):
        """Capture CUDA Graph for fixed batch size encoding (Optimization 1)"""
        dev = self._as_device(device)
        if not self._use_graphs or self._gr is not None or dev.type != "cuda":
            return
        
        # Check for no-op pool (Optimization 2)
        self._check_adaptive_pool(H, W, dev)
        
        # Fuse Conv+BN before capturing graph (Optimization 3)
        self._fuse_conv_bn()
        
        self.eval()
        self.to(device=dev, memory_format=torch.channels_last)
        torch.cuda.synchronize()
        
        # Create static input buffer
        static_in = torch.zeros(self._gr_bs, C, H, W, device=dev).to(memory_format=torch.channels_last)
        
        # Enable benchmark for warmup to find best algorithm (Optimization 6)
        old_benchmark = cudnn.benchmark
        cudnn.benchmark = True
        
        # Warmup runs (3x) to ensure cuDNN algo selection and cache warming
        with torch.inference_mode():
            if self._use_bf16:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    for _ in range(3):
                        y = self._adaptive_pool(self._resnet(static_in))
                        torch.cuda.synchronize()
                    # Get output shape for buffer allocation
                    y = self._adaptive_pool(self._resnet(static_in))
            else:
                for _ in range(3):
                    y = self._adaptive_pool(self._resnet(static_in))
                    torch.cuda.synchronize()
                # Get output shape for buffer allocation
                y = self._adaptive_pool(self._resnet(static_in))
        
        # Restore benchmark setting after warmup
        cudnn.benchmark = old_benchmark
        
        # Allocate static output buffer
        self._gr_out = torch.empty_like(y)  # same shape as output
        
        # Capture graph - inline operations without calling forward()
        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        
        if self._use_bf16:
            # Capture with BF16 autocast
            with torch.cuda.graph(g):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    y2 = self._adaptive_pool(self._resnet(static_in))
                self._gr_out.copy_(y2)  # write into static buffer (outside autocast)
        else:
            # Capture without autocast
            with torch.cuda.graph(g):
                y2 = self._adaptive_pool(self._resnet(static_in))  # inline, no .to() calls
                self._gr_out.copy_(y2)  # write into static buffer
        
        self._gr_in = static_in
        self._gr = g
        self._log().info(f"CUDA Graph captured for encoder with BS={self._gr_bs} after warmup")

    def graph_forward(self, x):
        """Execute captured CUDA Graph (x must be [32,3,H,W] channels_last)"""
        self._gr_in.copy_(x, non_blocking=True)
        self._gr.replay()
        return self._gr_out  # NCHW view onto static buffer

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: NCHW float tensor [B,3,H,W]. 
            Should already be in channels_last format from caller.
        Returns:
            NCHW tensor [B,256,enc_image_size,enc_image_size]
        """
        # Assume caller already provides channels_last input
        # (no conversion in hot path for graph compatibility)
        
        # Apply BF16 autocast if enabled (only for CUDA)
        if self._use_bf16 and images.device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                y = self._resnet(images)
                y = self._adaptive_pool(y)
        else:
            y = self._resnet(images)
            y = self._adaptive_pool(y)
        return y
