#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

import logging
import os

import torch
import torch.nn as nn
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
        torch.backends.cudnn.benchmark = True

        self.enc_image_size = enc_image_size
        self._encoder_dim = 256  # ResNet18 layer3 output channels

        # Keep the same module names/structure expected by the checkpoint:
        resnet = torchvision.models.resnet18(weights=None)
        trunk = list(resnet.children())[:-3]  # conv1..layer3
        self._resnet = nn.Sequential(*trunk)  # <-- name kept as _resnet

        self._adaptive_pool = nn.AdaptiveAvgPool2d((self.enc_image_size, self.enc_image_size))
        
        # CUDA Graphs support (Optimization 1) - ON by default
        self._use_graphs = bool(int(os.getenv("ENCODER_USE_GRAPHS", "1")))
        self._gr_bs = int(os.getenv("ENCODER_BLOCK_BS", "32"))
        self._gr_in = None
        self._gr_out = None
        self._gr = None
        self._gr_reuses_output = True  # caller must .clone() if they need a fresh tensor

    def _log(self):
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    def get_encoder_dim(self) -> int:
        return self._encoder_dim

    def _maybe_capture(self, C=3, H=448, W=448, device="cuda"):
        """Capture CUDA Graph for fixed batch size encoding (Optimization 1)"""
        if not self._use_graphs or self._gr is not None or device != "cuda":
            return
        
        self.eval()
        self.to(device=device, memory_format=torch.channels_last)
        torch.cuda.synchronize()
        
        # Create static input buffer
        static_in = torch.zeros(self._gr_bs, C, H, W, device=device).to(memory_format=torch.channels_last)
        
        # Run once to materialize shapes & cuDNN algo choices
        with torch.inference_mode():
            _ = self._resnet(static_in)
            _ = self._adaptive_pool(_)
        
        # Allocate static output buffer by running full forward
        with torch.inference_mode():
            y = self.forward(static_in)  # NCHW output
        self._gr_out = torch.empty_like(y)  # same NCHW shape
        
        # Capture graph
        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            y2 = self.forward(static_in)
            self._gr_out.copy_(y2)  # write into static buffer
        
        self._gr_in = static_in
        self._gr = g
        self._log().info(f"CUDA Graph captured for encoder with BS={self._gr_bs}")

    def graph_forward(self, x):
        """Execute captured CUDA Graph (x must be [32,3,H,W] channels_last)"""
        self._gr_in.copy_(x, non_blocking=True)
        self._gr.replay()
        return self._gr_out  # NCHW view onto static buffer

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: NCHW float tensor [B,3,H,W]. H,W fixed (e.g., 448x448).
        Returns:
            NCHW tensor [B,256,enc_image_size,enc_image_size]
        """
        assert images.dim() == 4 and images.size(1) == 3, "images should be [B,3,H,W]"
        x = images.to(memory_format=torch.channels_last)  # perf-safe, doesn't affect state_dict
        y = self._resnet(x)
        y = self._adaptive_pool(y)
        return y