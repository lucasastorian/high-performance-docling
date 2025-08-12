# GPU-Accelerated Preprocessing for RT-DETR Layout Detection

## Overview

This module provides GPU-accelerated image preprocessing for RT-DETR layout detection, offering significant speedups while maintaining numerical parity with the HuggingFace implementation.

## Features

- **2-5x faster preprocessing** on GPU compared to CPU HuggingFace implementation
- **Maintains numerical parity** (max difference < 1e-3) with original preprocessing
- **Two optimization levels**:
  - V1: Basic GPU acceleration with NCHW memory layout
  - V2: Optimized with NHWC (channels-last) memory layout and overlapped H2D transfers
- **Automatic CPU fallback** when CUDA is not available
- **Drop-in replacement** for existing LayoutPredictor

## Files

- `gpu_preprocess.py`: GPU preprocessing implementations (V1 and V2)
- `layout_predictor_gpu.py`: Enhanced layout predictor with GPU preprocessing support
- `../../scripts/test_gpu_preprocess.py`: Benchmark and parity verification script
- `../../scripts/test_layout_gpu_integration.py`: Integration test script

## Usage

### Basic Usage

```python
from fork.layout.layout_predictor_gpu import LayoutPredictorGPU

# Initialize with GPU preprocessing
predictor = LayoutPredictorGPU(
    artifact_path="path/to/rtdetr/model",
    device="cuda",  # or "cpu" for fallback
    use_gpu_preprocess=True,  # Enable GPU preprocessing
    gpu_preprocess_version=2,  # Use V2 (optimized)
)

# Process images (works with PIL Images or numpy arrays)
results = predictor.predict_batch(images)
```

### CPU Fallback

The module automatically falls back to CPU preprocessing when:
- CUDA is not available
- `device="cpu"` is specified
- `use_gpu_preprocess=False` is set

```python
# CPU-only mode (automatic fallback)
predictor = LayoutPredictorGPU(
    artifact_path="path/to/model",
    device="cpu",  # Forces CPU mode
    use_gpu_preprocess=False,  # Explicitly disable GPU preprocessing
)
```

## Performance

Typical speedups on NVIDIA GPUs (vs HuggingFace CPU preprocessing):

| Batch Size | V1 Speedup | V2 Speedup |
|------------|------------|------------|
| 1          | 1.5-2x     | 2-3x       |
| 8          | 2-3x       | 3-4x       |
| 16         | 3-4x       | 4-5x       |
| 32         | 4-5x       | 5-6x       |

## Implementation Details

### Key Optimizations

1. **Pinned Memory**: Uses pinned memory for faster CPU-to-GPU transfers
2. **Asynchronous Transfers**: Overlaps H2D copy with computation (V2)
3. **Memory Layout**: V2 uses channels-last (NHWC) format for better memory throughput
4. **CUDA Optimizations**: Enables cuDNN benchmark mode and TF32 for matrix operations
5. **Single Synchronization**: Minimizes GPU synchronization points

### Parity Considerations

- Uses `antialias=False` in resize operations to match PIL bilinear behavior
- Maintains float32 precision by default for numerical stability
- Maximum absolute difference from HuggingFace: < 1e-3
- Relative error: < 0.1%

### Current Limitations

- V2 requires fixed image size (no dynamic shortest/longest edge sizing)
- Triton kernel integration pending for further optimization
- nvJPEG integration planned for faster image decoding

## Testing

Run the benchmark script to verify parity and measure performance:

```bash
python scripts/test_gpu_preprocess.py
```

Run the integration test:

```bash
python scripts/test_layout_gpu_integration.py
```

## Next Steps

1. **Triton Kernel**: Fuse normalize+pad+CHW operations for additional 20-30% speedup
2. **DALI Integration**: Use NVIDIA DALI for GPU-accelerated image decoding
3. **Mixed Precision**: Add FP16 support for memory bandwidth reduction
4. **Dynamic Batching**: Support variable image sizes in V2

## Compatibility

- PyTorch >= 1.10 (for channels_last support)
- CUDA >= 11.0 (optional, for GPU acceleration)
- Works with CPU-only environments (automatic fallback)