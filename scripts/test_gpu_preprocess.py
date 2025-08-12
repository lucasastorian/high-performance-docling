#!/usr/bin/env python3
"""
Test script to verify GPU preprocessing parity with HF and benchmark performance.
"""

import torch
import numpy as np
from PIL import Image
import time
from typing import List, Tuple
import os
import sys

# Add fork directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import RTDetrImageProcessor
from fork.layout.gpu_preprocess import GPUPreprocessor, GPUPreprocessorV2
from fork.layout.layout_predictor import LayoutPredictor
from fork.layout.layout_predictor_gpu import LayoutPredictorGPU


def create_test_images(batch_size: int, fixed_size: bool = True) -> List[Image.Image]:
    """Create test images for benchmarking."""
    images = []
    for i in range(batch_size):
        if fixed_size:
            # Fixed 640x640 images
            h, w = 640, 640
        else:
            # Variable sizes for heterogeneous batch testing
            sizes = [(640, 480), (800, 600), (1024, 768), (512, 512)]
            h, w = sizes[i % len(sizes)]
        
        # Create random RGB image
        img_array = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        images.append(Image.fromarray(img_array))
    
    return images


def compare_tensors(t1: torch.Tensor, t2: torch.Tensor, name: str = "tensor") -> dict:
    """Compare two tensors and return statistics."""
    # Move to CPU for comparison
    t1_cpu = t1.cpu().float()
    t2_cpu = t2.cpu().float()
    
    # Compute differences
    abs_diff = torch.abs(t1_cpu - t2_cpu)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()
    
    # Relative difference (avoid division by zero)
    rel_diff = abs_diff / (torch.abs(t2_cpu) + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    return {
        "name": name,
        "shape_match": t1.shape == t2.shape,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
        "passes_1e-3": max_abs_diff <= 1e-3,
        "passes_1e-2": max_abs_diff <= 1e-2,
    }


def benchmark_preprocessing(batch_sizes: List[int] = [1, 8], num_warmup: int = 3, num_runs: int = 10):
    """Benchmark different preprocessing approaches."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()
    
    # Configuration matching RT-DETR defaults
    size_config = {"height": 640, "width": 640}
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # Initialize preprocessors
    hf_processor = RTDetrImageProcessor(
        size=size_config,
        do_resize=True,
        do_rescale=True,
        rescale_factor=1/255,
        do_normalize=True,
        image_mean=mean,
        image_std=std,
        do_pad=False,  # No padding for this test
    )
    
    gpu_v1 = GPUPreprocessor(
        size=size_config,
        do_pad=False,
        mean=mean,
        std=std,
        device=device,
        dtype=torch.float32,
    ) if device == "cuda" else None
    
    gpu_v2 = GPUPreprocessorV2(
        size=size_config,
        do_pad=False,
        mean=mean,
        std=std,
        device=device,
        dtype=torch.float32,
    ) if device == "cuda" else None
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n{'='*60}")
        print(f"Testing batch size: {batch_size}")
        print(f"{'='*60}")
        
        # Create test images
        test_images = create_test_images(batch_size, fixed_size=True)
        
        # Test HF CPU preprocessing
        print("\n1. HuggingFace CPU Preprocessing")
        print("-" * 40)
        
        # Warmup
        for _ in range(num_warmup):
            hf_output = hf_processor(images=test_images, return_tensors="pt")
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_start = time.perf_counter()
        for _ in range(num_runs):
            hf_output = hf_processor(images=test_images, return_tensors="pt")
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_hf = (time.perf_counter() - t_start) / num_runs * 1000
        
        hf_pixel_values = hf_output["pixel_values"]
        print(f"Time: {t_hf:.2f} ms")
        print(f"Output shape: {hf_pixel_values.shape}")
        print(f"Throughput: {batch_size / (t_hf / 1000):.1f} imgs/sec")
        
        # Store HF baseline for comparison
        hf_baseline = hf_pixel_values.to(device) if device == "cuda" else hf_pixel_values
        
        # Test GPU V1 preprocessing
        if gpu_v1:
            print("\n2. GPU Preprocessing V1 (Basic)")
            print("-" * 40)
            
            # Warmup
            for _ in range(num_warmup):
                gpu_v1_output = gpu_v1.preprocess_batch(test_images)
            
            # Benchmark
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            for _ in range(num_runs):
                gpu_v1_output = gpu_v1.preprocess_batch(test_images)
            torch.cuda.synchronize()
            t_v1 = (time.perf_counter() - t_start) / num_runs * 1000
            
            v1_pixel_values = gpu_v1_output["pixel_values"]
            print(f"Time: {t_v1:.2f} ms")
            print(f"Output shape: {v1_pixel_values.shape}")
            print(f"Throughput: {batch_size / (t_v1 / 1000):.1f} imgs/sec")
            print(f"Speedup vs HF: {t_hf / t_v1:.2f}x")
            
            # Compare with HF
            parity = compare_tensors(v1_pixel_values, hf_baseline, "V1 vs HF")
            print(f"\nParity check:")
            print(f"  Max abs diff: {parity['max_abs_diff']:.6f}")
            print(f"  Mean abs diff: {parity['mean_abs_diff']:.6f}")
            print(f"  Passes 1e-3 threshold: {'✓' if parity['passes_1e-3'] else '✗'}")
        
        # Test GPU V2 preprocessing
        if gpu_v2:
            print("\n3. GPU Preprocessing V2 (Optimized)")
            print("-" * 40)
            
            # Warmup
            for _ in range(num_warmup):
                gpu_v2_output = gpu_v2.preprocess_batch(test_images)
            
            # Benchmark
            torch.cuda.synchronize()
            t_start = time.perf_counter()
            for _ in range(num_runs):
                gpu_v2_output = gpu_v2.preprocess_batch(test_images)
            torch.cuda.synchronize()
            t_v2 = (time.perf_counter() - t_start) / num_runs * 1000
            
            v2_pixel_values = gpu_v2_output["pixel_values"]
            print(f"Time: {t_v2:.2f} ms")
            print(f"Output shape: {v2_pixel_values.shape}")
            print(f"Throughput: {batch_size / (t_v2 / 1000):.1f} imgs/sec")
            print(f"Speedup vs HF: {t_hf / t_v2:.2f}x")
            if gpu_v1:
                print(f"Speedup vs V1: {t_v1 / t_v2:.2f}x")
            
            # Compare with HF
            parity = compare_tensors(v2_pixel_values, hf_baseline, "V2 vs HF")
            print(f"\nParity check:")
            print(f"  Max abs diff: {parity['max_abs_diff']:.6f}")
            print(f"  Mean abs diff: {parity['mean_abs_diff']:.6f}")
            print(f"  Passes 1e-3 threshold: {'✓' if parity['passes_1e-3'] else '✗'}")
        
        # Store results
        result = {
            "batch_size": batch_size,
            "hf_ms": t_hf,
            "v1_ms": t_v1 if gpu_v1 else None,
            "v2_ms": t_v2 if gpu_v2 else None,
        }
        results.append(result)
    
    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Batch':<8} {'HF CPU':<12} {'GPU V1':<12} {'GPU V2':<12} {'V1 Speedup':<12} {'V2 Speedup':<12}")
    print("-" * 80)
    
    for r in results:
        hf_ms = f"{r['hf_ms']:.2f} ms"
        v1_ms = f"{r['v1_ms']:.2f} ms" if r['v1_ms'] else "N/A"
        v2_ms = f"{r['v2_ms']:.2f} ms" if r['v2_ms'] else "N/A"
        v1_speedup = f"{r['hf_ms'] / r['v1_ms']:.2f}x" if r['v1_ms'] else "N/A"
        v2_speedup = f"{r['hf_ms'] / r['v2_ms']:.2f}x" if r['v2_ms'] else "N/A"
        
        print(f"{r['batch_size']:<8} {hf_ms:<12} {v1_ms:<12} {v2_ms:<12} {v1_speedup:<12} {v2_speedup:<12}")


def test_full_pipeline():
    """Test the full layout prediction pipeline with GPU preprocessing."""
    print("\n" + "="*60)
    print("FULL PIPELINE TEST")
    print("="*60)
    
    # Check if model artifacts exist
    artifact_path = "fork/layout/rtdetr_model"  # Adjust path as needed
    if not os.path.exists(artifact_path):
        print(f"Warning: Model artifacts not found at {artifact_path}")
        print("Skipping full pipeline test")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create test images
    test_images = create_test_images(batch_size=4, fixed_size=True)
    
    # Initialize predictors
    print("\nInitializing predictors...")
    
    # Original CPU predictor
    cpu_predictor = LayoutPredictor(
        artifact_path=artifact_path,
        device=device,
        base_threshold=0.3,
    )
    
    # GPU-accelerated predictor V1
    gpu_predictor_v1 = LayoutPredictorGPU(
        artifact_path=artifact_path,
        device=device,
        base_threshold=0.3,
        use_gpu_preprocess=True,
        gpu_preprocess_version=1,
    ) if device == "cuda" else None
    
    # GPU-accelerated predictor V2
    gpu_predictor_v2 = LayoutPredictorGPU(
        artifact_path=artifact_path,
        device=device,
        base_threshold=0.3,
        use_gpu_preprocess=True,
        gpu_preprocess_version=2,
    ) if device == "cuda" else None
    
    # Run predictions
    print("\nRunning predictions...")
    
    # CPU baseline
    cpu_results = cpu_predictor.predict_batch(test_images)
    print(f"CPU predictor: {len(cpu_results)} images processed")
    print(f"  Preprocessing: {cpu_predictor._t_preprocess_ms:.1f} ms")
    print(f"  Predict: {cpu_predictor._t_predict_ms:.1f} ms")
    print(f"  Postprocess: {cpu_predictor._t_postprocess_ms:.1f} ms")
    
    if gpu_predictor_v1:
        v1_results = gpu_predictor_v1.predict_batch(test_images)
        print(f"\nGPU V1 predictor: {len(v1_results)} images processed")
        print(f"  Preprocessing: {gpu_predictor_v1._t_preprocess_ms:.1f} ms")
        print(f"  Predict: {gpu_predictor_v1._t_predict_ms:.1f} ms")
        print(f"  Postprocess: {gpu_predictor_v1._t_postprocess_ms:.1f} ms")
        print(f"  Speedup (preprocess): {cpu_predictor._t_preprocess_ms / gpu_predictor_v1._t_preprocess_ms:.2f}x")
    
    if gpu_predictor_v2:
        v2_results = gpu_predictor_v2.predict_batch(test_images)
        print(f"\nGPU V2 predictor: {len(v2_results)} images processed")
        print(f"  Preprocessing: {gpu_predictor_v2._t_preprocess_ms:.1f} ms")
        print(f"  Predict: {gpu_predictor_v2._t_predict_ms:.1f} ms")
        print(f"  Postprocess: {gpu_predictor_v2._t_postprocess_ms:.1f} ms")
        print(f"  Speedup (preprocess): {cpu_predictor._t_preprocess_ms / gpu_predictor_v2._t_preprocess_ms:.2f}x")


if __name__ == "__main__":
    # Run benchmarks
    benchmark_preprocessing(batch_sizes=[1, 8, 16, 32])
    
    # Test full pipeline if model is available
    test_full_pipeline()
    
    print("\n✅ Testing complete!")