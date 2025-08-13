import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn.utils.fusion import fuse_conv_bn_eval
import torchvision
import docling_ibm_models.tableformer.settings as s

LOG_LEVEL = 20  # INFO


class Encoder04(nn.Module):
    """
    ResNet-18 stem producing 256-channel feature maps at stride 16.
    Input:  [B,3,448,448] (channels_last)
    Output: [B,256,enc_image_size,enc_image_size] (NCHW, channels_last memory)
    """

    def __init__(self, enc_image_size: int):
        super().__init__()
        torch.backends.cuda.matmul.allow_tf32 = True
        cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        self.enc_image_size = enc_image_size
        self._encoder_dim = 256

        resnet = torchvision.models.resnet18(weights=None)
        trunk = list(resnet.children())[:-3]  # conv1..layer3 (stride 16)
        self._resnet = nn.Sequential(*trunk)

        # Pool only if needed; will be Identity for 448→28
        self._adaptive_pool = nn.AdaptiveAvgPool2d((enc_image_size, enc_image_size))

        # Flags
        self._fused = False
        self._pool_checked = False

        # Optional BF16
        self._use_bf16 = os.getenv("TABLE_ENCODER_BF16", "0") == "1"

        # Move parameters and future activations to channels_last
        self.to(device="cuda", memory_format=torch.channels_last)

    def _log(self):
        return s.get_custom_logger(self.__class__.__name__, LOG_LEVEL)

    def get_encoder_dim(self) -> int:
        return self._encoder_dim

    def prepare_for_inference(self, device="cuda", sample_shape=(3, 448, 448)):
        """
        Call after loading weights. Performs:
          - Conv+BN fusion
          - Replace no-op pool with Identity
          - cuDNN warmup and set benchmark=True
        """
        self.eval().to(device)
        if not self._fused:
            self._fuse_conv_bn()

        # Check once with the real runtime size
        if not self._pool_checked:
            C, H, W = sample_shape
            with torch.inference_mode():
                dummy = torch.zeros(1, C, H, W, device=device).contiguous(memory_format=torch.channels_last)
                out = self._resnet(dummy)
            if out.shape[-2:] == (self.enc_image_size, self.enc_image_size):
                self._adaptive_pool = nn.Identity()
                self._log().info("AdaptiveAvgPool2d is a no-op; replaced with Identity")
            self._pool_checked = True

        # Do a short warmup to settle cuDNN algo, then enable benchmark
        with torch.inference_mode():
            dummy = torch.zeros(4, *sample_shape, device=device).contiguous(memory_format=torch.channels_last)
            for _ in range(3):
                _ = self.forward(dummy)
            torch.cuda.synchronize()
        cudnn.benchmark = True  # fixed shapes → best perf
        return self

    def _fuse_conv_bn(self):
        self.eval()
        # Fuse conv1+bn1
        modules = dict(self._resnet.named_children())
        if '0' in modules and '1' in modules:
            conv1 = modules['0']; bn1 = modules['1']
            if isinstance(conv1, nn.Conv2d) and isinstance(bn1, nn.BatchNorm2d):
                fused_conv1 = fuse_conv_bn_eval(conv1, bn1)
                new_modules = []
                for name, module in self._resnet.named_children():
                    if name == '0': new_modules.append(fused_conv1)
                    elif name == '1': new_modules.append(nn.Identity())
                    else: new_modules.append(module)
                self._resnet = nn.Sequential(*new_modules)

        # Fuse BasicBlocks
        for m in self._resnet.modules():
            if isinstance(m, torchvision.models.resnet.BasicBlock):
                if hasattr(m, 'conv1') and hasattr(m, 'bn1'):
                    m.conv1 = fuse_conv_bn_eval(m.conv1, m.bn1); m.bn1 = nn.Identity()
                if hasattr(m, 'conv2') and hasattr(m, 'bn2'):
                    m.conv2 = fuse_conv_bn_eval(m.conv2, m.bn2); m.bn2 = nn.Identity()
                if m.downsample is not None and len(m.downsample) == 2:
                    dconv, dbn = m.downsample
                    if isinstance(dconv, nn.Conv2d) and isinstance(dbn, nn.BatchNorm2d):
                        m.downsample = nn.Sequential(fuse_conv_bn_eval(dconv, dbn), nn.Identity())
        self._fused = True
        self._log().info("Conv+BN fused")

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Expect channels_last input; do not change memory format here
        if images.device.type == "cuda":
            assert images.is_contiguous(memory_format=torch.channels_last), \
                "Pass encoder inputs contiguous(channels_last) before calling forward()"

        if self._use_bf16 and images.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                y = self._resnet(images)
                y = self._adaptive_pool(y)
        else:
            y = self._resnet(images)
            y = self._adaptive_pool(y)

        # y is NCHW Tensor; its storage follows channels_last activations
        return y
