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
        
        # Compilation support - OFF by default
        self._use_compile = bool(int(os.getenv("ENCODER_COMPILE", "0")))
        
        # CUDA Graphs support - ON by default (but disabled if compile is on)
        self._use_graphs = bool(int(os.getenv("ENCODER_USE_GRAPHS", "1"))) and not self._use_compile
        self._gr_bs = int(os.getenv("ENCODER_BLOCK_BS", "32"))
        self._gr_in = None
        self._gr_out = None
        self._gr = None
        self._gr_reuses_output = True  # caller must .clone() if they need a fresh tensor
        self._fused = False  # Track if Conv+BN fusion has been applied
        self._pool_checked = False  # Track if we've checked for no-op pool
        self._compiled = False  # Track if model has been compiled
        
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
    
    def prepare_for_inference(self, device="cuda"):
        """Call this AFTER loading weights to prepare model for inference.
        This handles fusion, compilation, and graph capture in the correct order.
        Returns self (or compiled wrapper if compilation is enabled).
        """
        dev = self._as_device(device)
        encoder = self  # Track the encoder object (may be replaced by compile)
        
        # Step 1: Fuse Conv+BN layers if not already done
        if not encoder._fused:
            encoder._fuse_conv_bn()
        
        # Step 2: Check if pool is no-op and replace with Identity BEFORE compile
        # This avoids compiling unnecessary ops
        if dev.type == "cuda":
            encoder._check_adaptive_pool(H=448, W=448, device=dev)
        
        # Step 3: Compile if requested (mutually exclusive with manual graphs)
        if encoder._use_compile and not encoder._compiled:
            encoder = encoder.compile_self(device=dev)  # Returns compiled wrapper
            
            # CRITICAL: Warm up the compiled model so first real inference isn't slow
            # This triggers torch.compile's graph compilation/JIT outside of timing
            if dev.type == "cuda":
                self._log().info("Warming up compiled model...")
                B = self._gr_bs
                C, H, W = 3, 448, 448  # Standard input dimensions
                dummy = torch.zeros(B, C, H, W, device=dev).contiguous(memory_format=torch.channels_last)
                torch.cuda.synchronize()
                
                # 2-3 warmup runs to trigger compilation and cache
                for i in range(3):
                    _ = encoder.forward(dummy)
                    torch.cuda.synchronize()
                
                self._log().info("Compiled model warmup complete")
        
        # Step 4: Capture CUDA graphs if requested (only if not compiled)
        elif encoder._use_graphs and encoder._gr is None and dev.type == "cuda":
            # Will internally call _fuse_conv_bn if needed
            encoder._maybe_capture(device=dev)
        
        self._log().info(f"Model prepared for inference - Fused: {encoder._fused}, Compiled: {encoder._compiled}, Graphs: {encoder._gr is not None}")
        
        return encoder  # Return the encoder (may be compiled wrapper)
    
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

    def compile_self(self, device="cuda"):
        """Compile the ENTIRE encoder module with torch.compile
        Returns the compiled module (since torch.compile returns a new wrapper)
        """
        if self._compiled:
            return self
        
        # Default to CUDA if not specified
        dev = self._as_device(device)
        if dev.type != "cuda":
            self._log().warning("torch.compile only supported on CUDA, skipping compilation")
            return self
        
        # Move to device and eval mode
        self.eval()
        self.to(device=dev)
        
        # ALWAYS use reduce-overhead for conv-heavy models like ResNet
        # This mode uses CUDA graphs internally for best performance
        compiled_encoder = torch.compile(
            self,  # Compile the ENTIRE encoder, not submodules!
            mode="reduce-overhead",  # Best for static shapes, uses CUDA graphs internally
            fullgraph=True,  # No fallback to eager mode
            dynamic=False  # Static shapes only (we always use same H,W,BS)
        )
        
        self._log().info("Encoder compiled with torch.compile (reduce-overhead mode)")
        self._compiled = True
        
        return compiled_encoder  # Return the compiled wrapper
    
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
        """Capture CUDA Graph for fixed batch size encoding (Optimization 1)
        
        IMPORTANT: H and W should match your actual runtime image dimensions
        to properly detect if adaptive pool is a no-op.
        """
        dev = self._as_device(device)
        if not self._use_graphs or self._gr is not None or dev.type != "cuda":
            return
        
        # Check for no-op pool with actual runtime dimensions (Optimization 2)
        self._check_adaptive_pool(H, W, dev)
        
        # Fuse Conv+BN before capturing graph (Optimization 3)
        self._fuse_conv_bn()
        
        self.eval()
        self.to(device=dev)  # Weights stay in default format, only activations are channels_last
        torch.cuda.synchronize()
        
        # Create static input buffer - ensure it's properly channels_last
        static_in = torch.zeros(self._gr_bs, C, H, W, device=dev).contiguous(memory_format=torch.channels_last)
        
        # Warmup runs (3x) to ensure cuDNN algo selection and cache warming
        # cudnn.benchmark stays at its default (False) to avoid per-shape algo search
        for _ in range(3):
            _ = self.forward(static_in)
            torch.cuda.synchronize()
        
        # Get output shape for buffer allocation
        y = self.forward(static_in)
        self._gr_out = torch.empty_like(y)
        
        # Capture graph using the same forward() path as runtime
        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        
        with torch.cuda.graph(g):
            y2 = self.forward(static_in)
            self._gr_out.copy_(y2)  # write into static buffer
        
        self._gr_in = static_in
        self._gr = g
        self._log().info(f"CUDA Graph captured for encoder with BS={self._gr_bs} after warmup")

    def get_graph_input_buffer(self) -> torch.Tensor:
        """Get the static input buffer for direct writing (avoids copy)"""
        return self._gr_in  # NHWC contiguous, shape [gr_bs, C, H, W]
    
    def graph_replay(self):
        """Replay the graph (assumes input buffer already filled)"""
        self._gr.replay()
        return self._gr_out
    
    def graph_forward(self, x):
        """Execute captured CUDA Graph (x must be [32,3,H,W] channels_last)"""
        assert x.is_contiguous(memory_format=torch.channels_last), "Input must be channels_last for graph replay"
        self._gr_in.copy_(x, non_blocking=True)
        self._gr.replay()
        return self._gr_out  # NCHW view onto static buffer

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: NCHW float tensor [B,3,H,W]. 
        Returns:
            NCHW tensor [B,256,enc_image_size,enc_image_size]
        """
        # Assert batch size is always 32 when graphs are enabled (manual graphs only)
        if self._use_graphs and images.device.type == "cuda":
            assert images.shape[0] == self._gr_bs, f"Batch size must be {self._gr_bs} for CUDA graphs, got {images.shape[0]}"
        
        # Fix 1: Ensure channels_last format (cheap if already channels_last)
        if images.device.type == "cuda" and not images.is_contiguous(memory_format=torch.channels_last):
            images = images.contiguous(memory_format=torch.channels_last)
        
        # Apply BF16 autocast if enabled (only for CUDA)
        if self._use_bf16 and images.device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                y = self._resnet(images)
                y = self._adaptive_pool(y)
        else:
            y = self._resnet(images)
            y = self._adaptive_pool(y)
        return y
