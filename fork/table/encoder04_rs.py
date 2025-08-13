# Copyright IBM Corp. 2024 - 2024
# SPDX-License-Identifier: MIT
#
import os
import logging
from typing import Optional

import torch
import torch.nn as nn
import torchvision

from torch.ao.quantization import fuse_modules
from torchvision.models.resnet import BasicBlock

import docling_ibm_models.tableformer.settings as s

LOG_LEVEL = logging.INFO


class Encoder04(nn.Module):
    """
    ResNet-18 stem producing 256-channel feature maps at stride 16.
    Optimized for inference:
      - Always eval() outside callers
      - Channels-last I/O and params for faster cuDNN kernels
      - cuDNN autotune enabled globally
      - Conv+BN (+ReLU where applicable) fused in eval
      - Optional CUDA Graphs fast-path for BS=32 (pad or pass exactly 32)
      - Optional torch.compile gated by TORCH_COMPILE=1

    Output (always): NCHW tensor of shape [B, 256, enc_image_size, enc_image_size]
    """

    def __init__(self, enc_image_size: int, enc_dim: int = 512, *, block_bs: int = 32):
        super().__init__()

        # --- global backend knobs (safe for inference) ---
        torch.backends.cudnn.benchmark = True  # fixed shapes => autotune helps

        self.enc_image_size = enc_image_size
        self._encoder_dim = 256         # ResNet18 layer3 output channels
        self._block_bs = int(block_bs)  # CUDA-graph micro-batch

        # Flags from env
        self._use_graph = os.getenv("USE_CUDAGRAPH", "1") == "1"
        self._use_compile = os.getenv("TORCH_COMPILE", "0") == "1"
        self._clone_graph_out = os.getenv("ENCODER_CLONE_GRAPH_OUTPUT", "0") == "1"

        # --- build resnet18 up to layer3 (stride 16, 256ch) ---
        resnet = torchvision.models.resnet18(weights=None)
        # conv1,bn1,relu,maxpool,layer1,layer2,layer3  -> yields 256xH/16xW/16
        trunk = list(resnet.children())[:-3]
        self._resnet = nn.Sequential(*trunk)

        # spatial resize head (kept separate so we can fuse/compile the whole path)
        self._adaptive_pool = nn.AdaptiveAvgPool2d((self.enc_image_size, self.enc_image_size))

        # Place modules in channels_last memory format (weights/buffers)
        self._resnet = self._resnet.to(memory_format=torch.channels_last)
        self._adaptive_pool = self._adaptive_pool.to(memory_format=torch.channels_last)

        # Try to fuse Conv+BN(+ReLU) in eval for speed (no numeric change)
        self._attempt_fuse_eval(self._resnet)

        # Combine into a single module for compile/graph friendliness
        self._graphable = nn.Sequential(self._resnet, self._adaptive_pool)

        # Optional compile (kept behind flag)
        if self._use_compile:
            # Good default; tweak if you want: mode="max-autotune" on 2.4+
            self._graphable = torch.compile(self._graphable, fullgraph=False, mode="reduce-overhead")

        # CUDA Graph runner (initialized lazily on first BS=block_bs call)
        self._gr = _CUDAGraphRunner(self._graphable,
                                    static_bs=self._block_bs,
                                    clone_output=self._clone_graph_out,
                                    enabled=self._use_graph)

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
        # Use channels-last for faster convs. If already channels_last, this is a no-op.
        x = images.to(memory_format=torch.channels_last)

        # Fast path: BS == static block and CUDA Graphs enabled + CUDA device
        y = self._gr.maybe_replay(x)
        if y is not None:
            return y  # [B=block_bs, 256, H', W']

        # Fallback: regular eager forward (still channels_last)
        y = self._graphable(x)  # [B,256,H',W']
        return y

    # -------- helpers --------

    @staticmethod
    def _attempt_fuse_eval(seq: nn.Sequential) -> None:
        """
        Fuse Conv+BN(+ReLU) where possible. Safe in eval only.
        The sequential layout after slicing resnet allows:
          - stem: [0]=conv1, [1]=bn1, [2]=relu  -> fuse(['0','1','2'])
          - inside BasicBlock: fuse(['conv1','bn1','relu']) and ['conv2','bn2']
            and downsample (if present): fuse(['0','1'])
        """
        try:
            seq.eval()
            # Fuse stem if it matches conv1/bn1/relu
            try:
                fuse_modules(seq, ['0', '1', '2'], inplace=True)
            except Exception:
                pass  # non-fatal if structure differs

            # Fuse residual blocks inside layer1..3
            for m in seq.modules():
                if isinstance(m, BasicBlock):
                    # main path
                    try:
                        fuse_modules(m, ['conv1', 'bn1', 'relu'], inplace=True)
                        fuse_modules(m, ['conv2', 'bn2'], inplace=True)
                    except Exception:
                        pass
                    # downsample path (Conv+BN)
                    if getattr(m, 'downsample', None) and isinstance(m.downsample, nn.Sequential):
                        # expect indices '0' (Conv), '1' (BN)
                        try:
                            fuse_modules(m.downsample, ['0', '1'], inplace=True)
                        except Exception:
                            pass
        except Exception:
            # Never fail hard on fusing: performance-only hint
            pass


class _CUDAGraphRunner:
    """
    Minimal CUDA Graph wrapper for a fixed batch size (static_bs).
    Captures: y = module(x) where x:[static_bs,3,H,W] (channels_last).
    Replays when forward receives BS==static_bs on CUDA in no-grad mode.

    Contract: Returned tensor points to a static buffer; valid until next replay.
              Set ENCODER_CLONE_GRAPH_OUTPUT=1 to return a cloned copy (slower).
    """
    def __init__(self, module: nn.Module, static_bs: int, clone_output: bool, enabled: bool):
        self.module = module
        self.static_bs = static_bs
        self.clone_output = clone_output
        self.enabled = enabled
        self.captured = False

        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_in: Optional[torch.Tensor] = None
        self.static_out: Optional[torch.Tensor] = None

    @torch.inference_mode()
    def maybe_replay(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        if not (self.enabled and x.is_cuda and not torch.is_grad_enabled()):
            return None
        if x.size(0) != self.static_bs:
            return None

        if not self.captured:
            self._warmup_and_capture(x)

        # Copy user input into static buffer (no shape change)
        self.static_in.copy_(x)
        self.graph.replay()  # type: ignore[attr-defined]
        return self.static_out.clone() if self.clone_output else self.static_out

    @torch.inference_mode()
    def _warmup_and_capture(self, example: torch.Tensor) -> None:
        dev = example.device
        # Allocate persistent static buffers with same dtype/format
        self.static_in = torch.empty_like(example, device=dev).to(memory_format=torch.channels_last)
        with torch.no_grad():
            # one dry run (algorithm selection, allocator warming)
            _ = self.module(self.static_in)

        # Allocate output buffer by running once deterministically
        self.static_out = self.module(self.static_in)

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        # Make sure we don't allocate new memory during capture
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph):
            self.static_out = self.module(self.static_in)
        torch.cuda.synchronize()
        self.captured = True
