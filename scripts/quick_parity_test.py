#!/usr/bin/env python3
"""
Quick parity and timing test for GPU preprocessing.
"""

import torch
import time
import numpy as np
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def bench_once(f, *a, **k):
    """Single benchmark run with proper synchronization."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    out = f(*a, **k)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.time() - t0) * 1000.0, out


def test_parity():
    """Quick parity test between HF and GPU preprocessing."""
    
    print("="*60)
    print("QUICK PARITY TEST")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Import preprocessors
    try:
        from transformers import RTDetrImageProcessor
        from fork.layout.gpu_preprocess import GPUPreprocessor, GPUPreprocessorV2
    except ImportError as e:
        print(f"Import error: {e}")
        return
    
    # Make two random RGB images (hetero sizes for testing)
    imgs = [
        Image.fromarray(np.random.randint(0, 255, (720, 1280, 3), np.uint8)),
        Image.fromarray(np.random.randint(0, 255, (800, 600, 3), np.uint8))
    ]
    
    # Config (standard RT-DETR settings)
    size_config = {"height": 640, "width": 640}
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    print("\nTest configuration:")
    print(f"  Size: {size_config}")
    print(f"  Mean: {mean}")
    print(f"  Std: {std}")
    print(f"  Test images: {len(imgs)} images")
    
    # HF baseline (CPU)
    print("\n1. HuggingFace CPU Preprocessing")
    print("-" * 40)
    # Note: RT-DETR typically uses do_normalize=False
    hf = RTDetrImageProcessor(
        size=size_config,
        do_resize=True,
        do_rescale=True,
        rescale_factor=1/255,
        do_normalize=False,  # RT-DETR default
        image_mean=mean,
        image_std=std,
        do_pad=False,
    )
    
    t_cpu, out_cpu = bench_once(lambda xs: hf(images=xs, return_tensors="pt"), imgs)
    x_cpu = out_cpu["pixel_values"]  # (B,C,H,W) float32
    print(f"  Time: {t_cpu:.1f} ms")
    print(f"  Output shape: {x_cpu.shape}")
    print(f"  Output device: {x_cpu.device}")
    
    if device == "cuda":
        # Test GPU V1
        print("\n2. GPU Preprocessing V1")
        print("-" * 40)
        gpu_v1 = GPUPreprocessor(
            size=size_config,
            do_pad=False,
            do_rescale=True,
            rescale_factor=1/255,
            do_normalize=False,  # Match HF config
            mean=mean,
            std=std,
            device="cuda:0",
            dtype=torch.float32
        )
        
        t_gpu_v1, out_gpu_v1 = bench_once(gpu_v1.preprocess_batch, imgs)
        x_gpu_v1 = out_gpu_v1["pixel_values"].cpu()
        
        print(f"  Time: {t_gpu_v1:.1f} ms")
        print(f"  Output shape: {x_gpu_v1.shape}")
        print(f"  Speedup: {t_cpu / t_gpu_v1:.2f}x")
        
        # Parity check V1
        diff_v1 = (x_cpu - x_gpu_v1).abs()
        max_diff_v1 = diff_v1.max().item()
        mean_diff_v1 = diff_v1.mean().item()
        print(f"\n  Parity (V1 vs HF):")
        print(f"    Max abs diff: {max_diff_v1:.4e}")
        print(f"    Mean abs diff: {mean_diff_v1:.4e}")
        print(f"    Passes 1e-3: {'✅' if max_diff_v1 <= 1e-3 else '❌'}")
        print(f"    Passes 1e-2: {'✅' if max_diff_v1 <= 1e-2 else '❌'}")
        
        # Test GPU V2
        print("\n3. GPU Preprocessing V2 (Optimized)")
        print("-" * 40)
        gpu_v2 = GPUPreprocessorV2(
            size=size_config,
            do_pad=False,
            do_rescale=True,
            rescale_factor=1/255,
            do_normalize=False,  # Match HF config
            mean=mean,
            std=std,
            device="cuda:0",
            dtype=torch.float32
        )
        
        t_gpu_v2, out_gpu_v2 = bench_once(gpu_v2.preprocess_batch, imgs)
        x_gpu_v2 = out_gpu_v2["pixel_values"].cpu()
        
        print(f"  Time: {t_gpu_v2:.1f} ms")
        print(f"  Output shape: {x_gpu_v2.shape}")
        print(f"  Speedup vs HF: {t_cpu / t_gpu_v2:.2f}x")
        print(f"  Speedup vs V1: {t_gpu_v1 / t_gpu_v2:.2f}x")
        
        # Parity check V2
        diff_v2 = (x_cpu - x_gpu_v2).abs()
        max_diff_v2 = diff_v2.max().item()
        mean_diff_v2 = diff_v2.mean().item()
        print(f"\n  Parity (V2 vs HF):")
        print(f"    Max abs diff: {max_diff_v2:.4e}")
        print(f"    Mean abs diff: {mean_diff_v2:.4e}")
        print(f"    Passes 1e-3: {'✅' if max_diff_v2 <= 1e-3 else '❌'}")
        print(f"    Passes 1e-2: {'✅' if max_diff_v2 <= 1e-2 else '❌'}")
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"HF CPU: {t_cpu:.1f} ms")
        if device == "cuda":
            print(f"GPU V1: {t_gpu_v1:.1f} ms (speedup: {t_cpu/t_gpu_v1:.2f}x)")
            print(f"GPU V2: {t_gpu_v2:.1f} ms (speedup: {t_cpu/t_gpu_v2:.2f}x)")
            print(f"\nParity target: max_abs_diff ≤ 1e-3")
            print(f"V1 parity: {'✅ PASS' if max_diff_v1 <= 1e-3 else f'⚠️  {max_diff_v1:.4e}'}")
            print(f"V2 parity: {'✅ PASS' if max_diff_v2 <= 1e-3 else f'⚠️  {max_diff_v2:.4e}'}")
    else:
        print("GPU not available - skipping GPU tests")
    
    print("\n✅ Test complete!")


def test_multi_gpu():
    """Test that multi-GPU device placement works correctly."""
    if not torch.cuda.is_available():
        print("CUDA not available - skipping multi-GPU test")
        return
    
    if torch.cuda.device_count() < 2:
        print(f"Only {torch.cuda.device_count()} GPU(s) available - skipping multi-GPU test")
        return
    
    print("\n" + "="*60)
    print("MULTI-GPU DEVICE PLACEMENT TEST")
    print("="*60)
    
    from fork.layout.gpu_preprocess import GPUPreprocessorV2
    
    # Test on cuda:0
    gpu0 = GPUPreprocessorV2(
        size={"height": 640, "width": 640},
        do_rescale=True,
        do_normalize=False,
        device="cuda:0",
        dtype=torch.float32
    )
    
    # Test on cuda:1
    gpu1 = GPUPreprocessorV2(
        size={"height": 640, "width": 640},
        do_rescale=True,
        do_normalize=False,
        device="cuda:1",
        dtype=torch.float32
    )
    
    # Create test image
    img = Image.fromarray(np.random.randint(0, 255, (640, 640, 3), np.uint8))
    
    # Process on GPU 0
    out0 = gpu0.preprocess_batch([img])
    print(f"GPU 0 output device: {out0['pixel_values'].device}")
    
    # Process on GPU 1
    out1 = gpu1.preprocess_batch([img])
    print(f"GPU 1 output device: {out1['pixel_values'].device}")
    
    # Verify devices
    assert str(out0['pixel_values'].device) == "cuda:0", "GPU 0 placement failed"
    assert str(out1['pixel_values'].device) == "cuda:1", "GPU 1 placement failed"
    
    print("✅ Multi-GPU placement test passed!")


if __name__ == "__main__":
    test_parity()
    test_multi_gpu()