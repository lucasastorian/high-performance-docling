import torch
import os
import contextlib
import torch.nn as nn

def device_from_str(s: str):
    s = str(s).lower()
    if s == "cuda" and torch.cuda.is_available(): 
        return torch.device("cuda")
    if s == "mps" and torch.backends.mps.is_available(): 
        return torch.device("mps")
    return torch.device("cpu")

def is_cuda(dev): 
    return dev.type == "cuda"

def is_mps(dev):  
    return dev.type == "mps"

def is_cpu(dev):  
    return dev.type == "cpu"

@contextlib.contextmanager
def safe_autocast(device, dtype=None):
    if is_cuda(device):
        with torch.autocast(device_type="cuda", dtype=dtype or torch.bfloat16):
            yield
    elif is_mps(device):
        # bfloat16 not fully supported on some MPS; use float16
        with torch.autocast(device_type="mps", dtype=dtype or torch.float16):
            yield
    elif is_cpu(device):
        # CPU autocast exists but can be slower; make it opt-in
        if os.environ.get("CPU_AUTOMIXED_PRECISION", "0") == "1":
            with torch.autocast(device_type="cpu", dtype=dtype or torch.bfloat16):
                yield
        else:
            yield
    else:
        yield

def enable_fast_backends():
    torch.backends.cuda.matmul.allow_tf32 = True if torch.cuda.is_available() else False
    # SDPA fastpaths (no-ops if backend missing)
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass
    # cuDNN autotune (CUDA only)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

def maybe_compile(mod, name="module"):
    try:
        return torch.compile(mod, mode="max-autotune")
    except Exception as e:
        print(f"[warn] torch.compile disabled for {name}: {e}")
        return mod

def prepare_model_for_infer(model, device):
    # channels_last helps on CPU & CUDA convolutions; OK on MPS
    model.to(memory_format=torch.channels_last, device=device)
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
            m.to(memory_format=torch.channels_last, device=device)
    return model

def to_device_images(img_batch, device):
    # img_batch: [B, 3, H, W] float32 0..1
    if is_cuda(device):
        # non_blocking requires pinned memory on host; pin if you own the tensor
        if not img_batch.is_pinned():
            img_batch = img_batch.pin_memory()
        return img_batch.to(device, non_blocking=True, memory_format=torch.channels_last)
    else:
        return img_batch.to(device, memory_format=torch.channels_last)