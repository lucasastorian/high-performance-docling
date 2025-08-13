#!/usr/bin/env python3
"""
Test Stage 1 batching: verify B=1 is identical and B=8 matches individual runs.
"""

import torch
import numpy as np
from fork.table.tablemodel04_rs import TableModel04_rs

def create_test_config():
    """Create minimal config for TableModel04_rs testing"""
    return {
        "model": {
            "enc_image_size": 28,
            "hidden_dim": 512,
            "tag_attention_dim": 256,
            "tag_embed_dim": 512,
            "tag_decoder_dim": 512,
            "bbox_attention_dim": 256,
            "bbox_embed_dim": 256,
            "bbox_classes": 3,
            "enc_layers": 6,
            "dec_layers": 6,
            "nheads": 8,
            "dropout": 0.1
        },
        "train": {
            "bbox": True
        },
        "predict": {
            "max_steps": 1024,
            "beam_size": 1,
            "profiling": False
        }
    }

def create_test_init_data():
    """Create minimal init_data for TableModel04_rs testing"""
    word_map_tag = {
        "<start>": 0,
        "<end>": 1,
        "<pad>": 2,
        "<unk>": 3,
        "fcel": 4,
        "ecel": 5,
        "ched": 6,
        "rhed": 7,
        "srow": 8,
        "nl": 9,
        "lcel": 10,
        "ucel": 11,
        "xcel": 12
    }
    
    return {
        "word_map": {
            "word_map_tag": word_map_tag
        }
    }

def test_stage1_batching():
    """Test Stage 1 batching implementation"""
    print("🧪 Testing Stage 1 Batching Implementation")
    print("=" * 50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    config = create_test_config()
    init_data = create_test_init_data()
    
    model = TableModel04_rs(config, init_data, device)
    model.eval()
    
    # Create test images: B=8, C=3, H=448, W=448
    torch.manual_seed(42)
    test_imgs = torch.randn(8, 3, 448, 448, device=device)
    max_steps = 50  # Short for testing
    k = 1
    
    print(f"Test images shape: {test_imgs.shape}")
    
    # Test 1: Single image (B=1) - baseline vs new implementation
    print("\n📋 Test 1: B=1 Baseline vs Stage1")
    img_single = test_imgs[0:1]
    
    # Baseline: Use original _predict_single method
    with torch.no_grad():
        seq_baseline, cls_baseline, coord_baseline = model._predict_single(
            img_single, max_steps, k
        )
    
    # Stage 1: Use new predict method with B=1
    with torch.no_grad():
        results_stage1 = model.predict(img_single, max_steps, k)
        seq_stage1, cls_stage1, coord_stage1 = results_stage1[0]
    
    # Compare results
    seqs_match = seq_baseline == seq_stage1
    print(f"  Sequences match: {seqs_match}")
    if not seqs_match:
        print(f"    Baseline: {seq_baseline[:20]}...")
        print(f"    Stage1:   {seq_stage1[:20]}...")
    
    # Compare bboxes (allowing small floating point differences)
    if cls_baseline is not None and cls_stage1 is not None:
        cls_close = torch.allclose(cls_baseline, cls_stage1, atol=1e-6)
        coord_close = torch.allclose(coord_baseline, coord_stage1, atol=1e-6)
        print(f"  Classes close: {cls_close}")
        print(f"  Coords close: {coord_close}")
    else:
        print(f"  Bbox outputs: baseline={cls_baseline is not None}, stage1={cls_stage1 is not None}")
    
    # Test 2: Batch (B=8) - each item should match individual baseline runs
    print("\n📋 Test 2: B=8 Batch vs Individual Baselines")
    
    # Get individual baseline results
    baseline_results = []
    with torch.no_grad():
        for i in range(8):
            seq, cls, coord = model._predict_single(
                test_imgs[i:i+1], max_steps, k
            )
            baseline_results.append((seq, cls, coord))
    
    # Get batched Stage 1 results
    with torch.no_grad():
        stage1_results = model.predict(test_imgs, max_steps, k)
    
    # Compare each item
    all_match = True
    for i in range(8):
        seq_base, cls_base, coord_base = baseline_results[i]
        seq_s1, cls_s1, coord_s1 = stage1_results[i]
        
        seq_match = seq_base == seq_s1
        if cls_base is not None and cls_s1 is not None:
            cls_match = torch.allclose(cls_base, cls_s1, atol=1e-6)
            coord_match = torch.allclose(coord_base, coord_s1, atol=1e-6)
            item_match = seq_match and cls_match and coord_match
        else:
            item_match = seq_match
        
        print(f"  Item {i}: {'✓' if item_match else '✗'}")
        if not item_match:
            all_match = False
            if not seq_match:
                print(f"    Seq mismatch: base len={len(seq_base)}, stage1 len={len(seq_s1)}")
    
    print(f"\n🎯 Overall batch consistency: {'✓ PASS' if all_match else '✗ FAIL'}")
    
    # Test 3: Performance comparison
    print("\n⏱️  Test 3: Performance Comparison")
    
    # Warmup
    with torch.no_grad():
        _ = model.predict(test_imgs[:2], max_steps, k)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Time individual runs
    import time
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(8):
            _ = model._predict_single(test_imgs[i:i+1], max_steps, k)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_individual = time.perf_counter() - start
    
    # Time batched run
    start = time.perf_counter()
    with torch.no_grad():
        _ = model.predict(test_imgs, max_steps, k)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time_batched = time.perf_counter() - start
    
    speedup = time_individual / time_batched if time_batched > 0 else 0
    print(f"  Individual (8x): {time_individual:.3f}s")
    print(f"  Batched (1x):    {time_batched:.3f}s") 
    print(f"  Speedup:         {speedup:.2f}x")
    
    return all_match

if __name__ == "__main__":
    success = test_stage1_batching()
    print(f"\n{'🎉 SUCCESS' if success else '❌ FAILED'}: Stage 1 batching test")