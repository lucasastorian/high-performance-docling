#
# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#

import logging

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
        # Do NOT fuse or wrap into another Sequential before weights load.

    def _log(self):
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    def get_encoder_dim(self) -> int:
        return self._encoder_dim

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