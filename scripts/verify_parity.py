#!/usr/bin/env python3
"""
Comprehensive parity verification for GPU preprocessing.
Tests exact tensor matching between HF and GPU paths.
"""

import torch
import numpy as np
from PIL import Image
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def verify_exact_parity(artifact_path: str = None):
    """Verify exact parity between HF and GPU preprocessing."""
    
    print("="*70)
    print("COMPREHENSIVE PARITY VERIFICATION")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    from transformers import RTDetrImageProcessor
    from fork.layout.gpu_preprocess import GPUPreprocessor, GPUPreprocessorV2
    
    # Create test images with different sizes
    test_images = [
        Image.fromarray(np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)),
        Image.fromarray(np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)),
        Image.fromarray(np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)),
    ]
    
    print(f"Test images: {len(test_images)} images with varying sizes")
    
    # Initialize HF processor
    if artifact_path and os.path.exists(os.path.join(artifact_path, "preprocessor_config.json")):
        print(f"\nLoading HF processor from: {artifact_path}")
        hf = RTDetrImageProcessor.from_pretrained(artifact_path)
        
        # Show actual config
        print(f"\nActual HF config:")
        print(f"  do_resize: {hf.do_resize}")
        print(f"  do_rescale: {hf.do_rescale}")
        print(f"  rescale_factor: {hf.rescale_factor}")
        print(f"  do_normalize: {hf.do_normalize}")
        print(f"  size: {hf.size}")
    else:
        print("\nUsing default RT-DETR config")
        hf = RTDetrImageProcessor(
            size={"height": 640, "width": 640},
            do_resize=True,
            do_rescale=True,
            rescale_factor=1/255,
            do_normalize=False,  # RT-DETR default
            do_pad=False,
        )
    
    # Process with HF
    print("\n" + "-"*70)
    print("HuggingFace Preprocessing")
    print("-"*70)
    
    cpu_out = hf(images=test_images, return_tensors="pt")
    cpu_tensor = cpu_out.pixel_values  # (B,C,H,W) float32
    
    print(f"Output shape: {cpu_tensor.shape}")
    print(f"Output dtype: {cpu_tensor.dtype}")
    print(f"Value range: [{cpu_tensor.min().item():.4f}, {cpu_tensor.max().item():.4f}]")
    print(f"Mean: {cpu_tensor.mean().item():.4f}")
    print(f"Std: {cpu_tensor.std().item():.4f}")
    
    if device == "cpu":
        print("\n⚠️  GPU not available - skipping GPU tests")
        return
    
    # Test GPU V1
    print("\n" + "-"*70)
    print("GPU Preprocessor V1")
    print("-"*70)
    
    gpu_v1 = GPUPreprocessor(
        size=hf.size,
        do_pad=hf.do_pad,
        pad_size=hf.pad_size if hasattr(hf, 'pad_size') else None,
        do_rescale=hf.do_rescale,
        rescale_factor=hf.rescale_factor,
        do_normalize=hf.do_normalize,
        mean=tuple(hf.image_mean),
        std=tuple(hf.image_std),
        device=str(device),
        dtype=torch.float32,
    )
    
    gpu_v1_out = gpu_v1.preprocess_batch(test_images)
    gpu_v1_tensor = gpu_v1_out["pixel_values"].cpu()
    
    print(f"Output shape: {gpu_v1_tensor.shape}")
    print(f"Output dtype: {gpu_v1_tensor.dtype}")
    print(f"Value range: [{gpu_v1_tensor.min().item():.4f}, {gpu_v1_tensor.max().item():.4f}]")
    print(f"Mean: {gpu_v1_tensor.mean().item():.4f}")
    print(f"Std: {gpu_v1_tensor.std().item():.4f}")
    
    # Compare V1
    diff_v1 = (cpu_tensor - gpu_v1_tensor).abs()
    print(f"\nParity (V1 vs HF):")
    print(f"  Max abs diff: {diff_v1.max().item():.6e}")
    print(f"  Mean abs diff: {diff_v1.mean().item():.6e}")
    print(f"  99th percentile: {torch.quantile(diff_v1, 0.99).item():.6e}")
    print(f"  95th percentile: {torch.quantile(diff_v1, 0.95).item():.6e}")
    
    v1_pass = diff_v1.max().item() <= 1e-3
    print(f"  Result: {'✅ PASS' if v1_pass else '❌ FAIL'} (target: max_abs ≤ 1e-3)")
    
    # Test GPU V2
    print("\n" + "-"*70)
    print("GPU Preprocessor V2 (Optimized)")
    print("-"*70)
    
    gpu_v2 = GPUPreprocessorV2(
        size=hf.size,
        do_pad=hf.do_pad,
        pad_size=hf.pad_size if hasattr(hf, 'pad_size') else None,
        do_rescale=hf.do_rescale,
        rescale_factor=hf.rescale_factor,
        do_normalize=hf.do_normalize,
        mean=tuple(hf.image_mean),
        std=tuple(hf.image_std),
        device=str(device),
        dtype=torch.float32,
    )
    
    gpu_v2_out = gpu_v2.preprocess_batch(test_images)
    gpu_v2_tensor = gpu_v2_out["pixel_values"].cpu()
    
    print(f"Output shape: {gpu_v2_tensor.shape}")
    print(f"Output dtype: {gpu_v2_tensor.dtype}")
    print(f"Value range: [{gpu_v2_tensor.min().item():.4f}, {gpu_v2_tensor.max().item():.4f}]")
    print(f"Mean: {gpu_v2_tensor.mean().item():.4f}")
    print(f"Std: {gpu_v2_tensor.std().item():.4f}")
    
    # Compare V2
    diff_v2 = (cpu_tensor - gpu_v2_tensor).abs()
    print(f"\nParity (V2 vs HF):")
    print(f"  Max abs diff: {diff_v2.max().item():.6e}")
    print(f"  Mean abs diff: {diff_v2.mean().item():.6e}")
    print(f"  99th percentile: {torch.quantile(diff_v2, 0.99).item():.6e}")
    print(f"  95th percentile: {torch.quantile(diff_v2, 0.95).item():.6e}")
    
    v2_pass = diff_v2.max().item() <= 1e-3
    print(f"  Result: {'✅ PASS' if v2_pass else '❌ FAIL'} (target: max_abs ≤ 1e-3)")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nTarget parity levels:")
    print("  Excellent: max_abs ≤ 1e-4")
    print("  Good: max_abs ≤ 1e-3")
    print("  Acceptable: max_abs ≤ 1e-2")
    
    print("\nResults:")
    if v1_pass and v2_pass:
        print("  ✅ Both V1 and V2 achieve good parity (≤ 1e-3)")
    elif v1_pass:
        print("  ⚠️  V1 passes but V2 needs adjustment")
    elif v2_pass:
        print("  ⚠️  V2 passes but V1 needs adjustment")
    else:
        print("  ❌ Both versions need adjustment")
    
    print("\nRecommendations:")
    if not hf.do_normalize:
        print("  • Model uses do_normalize=False (RT-DETR default)")
        print("  • Values should be in [0, 1] range after preprocessing")
    else:
        print("  • Model uses normalization")
        print("  • Values will be roughly in [-2, 2] range")
    
    if diff_v1.max().item() > 1e-3 or diff_v2.max().item() > 1e-3:
        print("\nTroubleshooting high diff:")
        print("  1. Check resize interpolation (should use antialias=False)")
        print("  2. Verify do_normalize/do_rescale flags match")
        print("  3. Ensure RGB channel order is consistent")
        print("  4. Check for any remaining GPU sync in rescale logic")


def test_detection_stability(artifact_path: str, threshold: float = 0.3):
    """Test that small preprocessing differences don't affect detections."""
    
    print("\n" + "="*70)
    print("DETECTION STABILITY TEST")
    print("="*70)
    
    if not artifact_path or not os.path.exists(artifact_path):
        print("⚠️  Model path required for detection test")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    from fork.layout.layout_predictor_gpu import LayoutPredictor
    
    # Create test image
    test_image = Image.fromarray(
        np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    )
    
    print(f"\nTesting with threshold: {threshold}")
    
    # Test with CPU preprocessing (baseline)
    predictor_cpu = LayoutPredictor(
        artifact_path=artifact_path,
        device=device,
        base_threshold=threshold,
        use_gpu_preprocess=False,  # Use HF CPU preprocessing
    )
    
    results_cpu = predictor_cpu.predict_batch([test_image])
    
    print(f"\nCPU preprocessing:")
    print(f"  Detections: {len(results_cpu[0])}")
    print(f"  Preprocessing time: {predictor_cpu._t_preprocess_ms:.1f} ms")
    
    if device == "cuda":
        # Test with GPU preprocessing
        predictor_gpu = LayoutPredictor(
            artifact_path=artifact_path,
            device=device,
            base_threshold=threshold,
            use_gpu_preprocess=True,
            gpu_preprocess_version=2,
        )
        
        results_gpu = predictor_gpu.predict_batch([test_image])
        
        print(f"\nGPU preprocessing:")
        print(f"  Detections: {len(results_gpu[0])}")
        print(f"  Preprocessing time: {predictor_gpu._t_preprocess_ms:.1f} ms")
        print(f"  Speedup: {predictor_cpu._t_preprocess_ms / predictor_gpu._t_preprocess_ms:.2f}x")
        
        # Compare detections
        cpu_count = len(results_cpu[0])
        gpu_count = len(results_gpu[0])
        
        if cpu_count == gpu_count:
            print(f"\n✅ Detection count matches: {cpu_count}")
        else:
            print(f"\n⚠️  Detection count differs: CPU={cpu_count}, GPU={gpu_count}")
            print("    This may indicate threshold sensitivity")
            
            # Try with slightly lower threshold
            lower_threshold = threshold - 0.02
            print(f"\n  Testing with lower threshold: {lower_threshold}")
            
            predictor_cpu_low = LayoutPredictor(
                artifact_path=artifact_path,
                device=device,
                base_threshold=lower_threshold,
                use_gpu_preprocess=False,
            )
            
            predictor_gpu_low = LayoutPredictor(
                artifact_path=artifact_path,
                device=device,
                base_threshold=lower_threshold,
                use_gpu_preprocess=True,
                gpu_preprocess_version=2,
            )
            
            results_cpu_low = predictor_cpu_low.predict_batch([test_image])
            results_gpu_low = predictor_gpu_low.predict_batch([test_image])
            
            print(f"    CPU: {len(results_cpu_low[0])} detections")
            print(f"    GPU: {len(results_gpu_low[0])} detections")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="Path to RT-DETR model artifacts")
    parser.add_argument("--threshold", type=float, default=0.3, help="Detection threshold")
    parser.add_argument("--test-detection", action="store_true", help="Test detection stability")
    args = parser.parse_args()
    
    # Run comprehensive parity test
    verify_exact_parity(args.model_path)
    
    # Run detection stability test if requested
    if args.test_detection and args.model_path:
        test_detection_stability(args.model_path, args.threshold)
    
    print("\n✅ Verification complete!")