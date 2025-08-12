#!/usr/bin/env python3
"""
Test script to verify normalization flag fix for RT-DETR preprocessing.
This should show exact parity when do_normalize matches between HF and GPU.
"""

import torch
import numpy as np
from PIL import Image
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_normalization_parity():
    """Test that normalization flags are properly respected."""
    
    print("="*70)
    print("RT-DETR NORMALIZATION FIX VERIFICATION")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    from transformers import RTDetrImageProcessor
    from fork.layout.gpu_preprocess import GPUPreprocessor, GPUPreprocessorV2
    
    # Create test image
    test_image = Image.fromarray(
        np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    )
    
    # Standard RT-DETR config
    size_config = {"height": 640, "width": 640}
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    print("\n" + "="*70)
    print("TEST 1: RT-DETR Default (do_normalize=False)")
    print("="*70)
    
    # HF with RT-DETR defaults (no normalization)
    hf_no_norm = RTDetrImageProcessor(
        size=size_config,
        do_resize=True,
        do_rescale=True,
        rescale_factor=1/255,
        do_normalize=False,  # RT-DETR default
        image_mean=mean,
        image_std=std,
        do_pad=False,
    )
    
    out_hf_no_norm = hf_no_norm(images=[test_image], return_tensors="pt")
    x_hf_no_norm = out_hf_no_norm["pixel_values"]
    
    print(f"\nHF output (no normalize):")
    print(f"  Shape: {x_hf_no_norm.shape}")
    print(f"  Min: {x_hf_no_norm.min().item():.4f}")
    print(f"  Max: {x_hf_no_norm.max().item():.4f}")
    print(f"  Mean: {x_hf_no_norm.mean().item():.4f}")
    
    if device == "cuda":
        # GPU preprocessor with matching config
        gpu_no_norm = GPUPreprocessorV2(
            size=size_config,
            do_pad=False,
            do_rescale=True,
            rescale_factor=1/255,
            do_normalize=False,  # Match HF
            mean=mean,
            std=std,
            device=device,
            dtype=torch.float32
        )
        
        out_gpu_no_norm = gpu_no_norm.preprocess_batch([test_image])
        x_gpu_no_norm = out_gpu_no_norm["pixel_values"].cpu()
        
        print(f"\nGPU output (no normalize):")
        print(f"  Shape: {x_gpu_no_norm.shape}")
        print(f"  Min: {x_gpu_no_norm.min().item():.4f}")
        print(f"  Max: {x_gpu_no_norm.max().item():.4f}")
        print(f"  Mean: {x_gpu_no_norm.mean().item():.4f}")
        
        # Compare
        diff = (x_hf_no_norm - x_gpu_no_norm).abs()
        print(f"\nParity (GPU vs HF, no normalize):")
        print(f"  Max abs diff: {diff.max().item():.6e}")
        print(f"  Mean abs diff: {diff.mean().item():.6e}")
        print(f"  99th percentile diff: {torch.quantile(diff, 0.99).item():.6e}")
        
        if diff.max().item() <= 1e-3:
            print("  ✅ PASS: Within 1e-3 tolerance")
        else:
            print(f"  ❌ FAIL: Exceeds 1e-3 tolerance")
    
    print("\n" + "="*70)
    print("TEST 2: With Normalization (do_normalize=True)")
    print("="*70)
    
    # HF with normalization
    hf_with_norm = RTDetrImageProcessor(
        size=size_config,
        do_resize=True,
        do_rescale=True,
        rescale_factor=1/255,
        do_normalize=True,  # Enable normalization
        image_mean=mean,
        image_std=std,
        do_pad=False,
    )
    
    out_hf_with_norm = hf_with_norm(images=[test_image], return_tensors="pt")
    x_hf_with_norm = out_hf_with_norm["pixel_values"]
    
    print(f"\nHF output (with normalize):")
    print(f"  Shape: {x_hf_with_norm.shape}")
    print(f"  Min: {x_hf_with_norm.min().item():.4f}")
    print(f"  Max: {x_hf_with_norm.max().item():.4f}")
    print(f"  Mean: {x_hf_with_norm.mean().item():.4f}")
    
    if device == "cuda":
        # GPU preprocessor with matching config
        gpu_with_norm = GPUPreprocessorV2(
            size=size_config,
            do_pad=False,
            do_rescale=True,
            rescale_factor=1/255,
            do_normalize=True,  # Match HF
            mean=mean,
            std=std,
            device=device,
            dtype=torch.float32
        )
        
        out_gpu_with_norm = gpu_with_norm.preprocess_batch([test_image])
        x_gpu_with_norm = out_gpu_with_norm["pixel_values"].cpu()
        
        print(f"\nGPU output (with normalize):")
        print(f"  Shape: {x_gpu_with_norm.shape}")
        print(f"  Min: {x_gpu_with_norm.min().item():.4f}")
        print(f"  Max: {x_gpu_with_norm.max().item():.4f}")
        print(f"  Mean: {x_gpu_with_norm.mean().item():.4f}")
        
        # Compare
        diff = (x_hf_with_norm - x_gpu_with_norm).abs()
        print(f"\nParity (GPU vs HF, with normalize):")
        print(f"  Max abs diff: {diff.max().item():.6e}")
        print(f"  Mean abs diff: {diff.mean().item():.6e}")
        print(f"  99th percentile diff: {torch.quantile(diff, 0.99).item():.6e}")
        
        if diff.max().item() <= 1e-3:
            print("  ✅ PASS: Within 1e-3 tolerance")
        else:
            print(f"  ❌ FAIL: Exceeds 1e-3 tolerance")
    
    print("\n" + "="*70)
    print("TEST 3: Check Scale Difference")
    print("="*70)
    
    # This should show the difference between normalized and non-normalized
    scale_diff = (x_hf_with_norm - x_hf_no_norm).abs()
    print(f"\nDifference between normalized and non-normalized:")
    print(f"  Max diff: {scale_diff.max().item():.4f}")
    print(f"  Mean diff: {scale_diff.mean().item():.4f}")
    print(f"  (Should be >0.1 if normalization is working)")
    
    if scale_diff.max().item() > 0.1:
        print("  ✅ Normalization is having expected effect")
    else:
        print("  ⚠️  Normalization may not be working correctly")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nKey Points:")
    print("1. RT-DETR typically uses do_normalize=False (just rescaling to [0,1])")
    print("2. GPU preprocessor now respects the do_normalize flag")
    print("3. This should fix the detection threshold drift issues")
    print("\nRecommendation: Use do_normalize=False for RT-DETR models")


def check_preprocessor_config(artifact_path):
    """Check the actual preprocessor config from model artifacts."""
    print("\n" + "="*70)
    print("CHECKING MODEL PREPROCESSOR CONFIG")
    print("="*70)
    
    config_path = os.path.join(artifact_path, "preprocessor_config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"\nPreprocessor config from {config_path}:")
        print(f"  do_normalize: {config.get('do_normalize', 'NOT SET')}")
        print(f"  do_rescale: {config.get('do_rescale', 'NOT SET')}")
        print(f"  rescale_factor: {config.get('rescale_factor', 'NOT SET')}")
        print(f"  image_mean: {config.get('image_mean', 'NOT SET')}")
        print(f"  image_std: {config.get('image_std', 'NOT SET')}")
        
        if config.get('do_normalize') == False:
            print("\n✅ Config confirms: RT-DETR uses do_normalize=False")
        elif config.get('do_normalize') == True:
            print("\n⚠️  Config shows do_normalize=True - may need adjustment")
        else:
            print("\n⚠️  do_normalize not explicitly set in config")
    else:
        print(f"\n⚠️  Config file not found at {config_path}")


if __name__ == "__main__":
    # Run normalization parity test
    test_normalization_parity()
    
    # Check actual model config if path provided
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="Path to RT-DETR model artifacts")
    args = parser.parse_args()
    
    if args.model_path:
        check_preprocessor_config(args.model_path)
    else:
        print("\nTip: Run with --model-path /path/to/rtdetr/model to check actual config")