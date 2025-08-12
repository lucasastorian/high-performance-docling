# RT-DETR Normalization Fix

## Problem Identified

The GPU preprocessing was **always normalizing** the images (applying `(x - mean) / std`), but RT-DETR models typically run with `do_normalize=False`, only rescaling to [0,1]. This mismatch caused:

1. **Numeric drift** in model inputs (different scale)
2. **Detection threshold fragility** - some boxes falling just below/above 0.3 threshold
3. **Layout differences** - ADDED/REMOVED tables, shifted bounding boxes
4. **IoU mismatches** in regression tests

## Root Cause

```python
# GPU preprocessor was doing:
batch_normalized = (batch_resized - self.mean) / self.std  # ALWAYS

# But HF RT-DETR defaults to:
do_normalize=False  # Just rescale to [0,1], no normalization
```

This scale difference (~0.2-1.0) was enough to shift model logits and change which detections survived thresholding.

## Solution Implemented

### 1. Added Normalization Flags

Both GPU preprocessors now accept and respect:
- `do_rescale`: Whether to scale pixels to [0,1]
- `rescale_factor`: Scale factor (typically 1/255)
- `do_normalize`: Whether to apply mean/std normalization

### 2. Pass Flags from HF Config

```python
self._gpu_preprocessor = PreprocessorClass(
    ...
    do_rescale=self._image_preprocessor.do_rescale,
    rescale_factor=self._image_preprocessor.rescale_factor,
    do_normalize=self._image_preprocessor.do_normalize,  # Respect HF config
    ...
)
```

### 3. Conditional Processing

```python
# Only rescale if needed
if self.do_rescale:
    batch_gpu = batch_gpu * self.rescale_factor

# Only normalize if needed  
if self.do_normalize:
    batch_normalized = (batch_resized - self.mean) / self.std
else:
    batch_normalized = batch_resized  # RT-DETR path
```

## Verification

Run the test script to verify parity:

```bash
python scripts/test_normalization_fix.py

# Check your model's actual config:
python scripts/test_normalization_fix.py --model-path /path/to/rtdetr/model
```

Expected results:
- **Without normalization**: Values in [0, 1] range
- **With normalization**: Values roughly in [-2, 2] range
- **Parity**: Max abs diff ≤ 1e-3 between HF and GPU

## Impact

This fix should resolve:
- ✅ Detection threshold drift
- ✅ False ADDED/REMOVED tables in regression
- ✅ Shifted bounding boxes (like 281.72 → 240.22)
- ✅ IoU mismatches in layout regression

## Configuration Check

To verify your RT-DETR model's config:

```python
import json
with open("path/to/model/preprocessor_config.json") as f:
    config = json.load(f)
    print(f"do_normalize: {config.get('do_normalize')}")  # Should be False
```

Most RT-DETR models use:
- `do_resize`: True
- `do_rescale`: True (to [0,1])
- `do_normalize`: **False** (no mean/std normalization)

## Next Steps

1. Run layout detection with fixed preprocessing
2. Verify regression tests pass
3. If still seeing small diffs, consider:
   - Slightly lower threshold (0.28 instead of 0.3)
   - IoU-based matching instead of exact bbox comparison