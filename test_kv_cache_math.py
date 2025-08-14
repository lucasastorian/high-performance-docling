#!/usr/bin/env python3
"""
Step 2: Prove self-attention single-step math equals full MHA
This is a standalone test to verify KV cache math before implementing in the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def check_sa_incremental_equiv(
    mha: nn.MultiheadAttention,
    T: int = 7,
    B: int = 3,
    E: int = 512,
    heads: int = 8,
    atol: float = 1e-6,
    device: str = 'cuda'
) -> float:
    """
    Verify that incremental KV cache produces identical results to full attention.
    
    Returns max absolute error between the two approaches.
    """
    mha = mha.to(device).eval()
    Dh = E // heads
    
    # Create random input sequence
    x = torch.randn(T, B, E, device=device)
    
    # ========== Method 1: Stock MHA (ground truth) ==========
    # Query from last token, attend to full sequence
    last = x[-1:, :, :]  # [1, B, E]
    full = x              # [T, B, E]
    
    with torch.no_grad():
        y_stock = mha(last, full, full, need_weights=False)[0]  # [1, B, E]
    
    # ========== Method 2: Manual KV cache ==========
    # Extract weights from MHA module
    W = mha.in_proj_weight  # [3*E, E]
    b = mha.in_proj_bias    # [3*E] or None
    
    W_q = W[:E, :]
    W_k = W[E:2*E, :]
    W_v = W[2*E:, :]
    b_q = None if b is None else b[:E]
    b_k = None if b is None else b[E:2*E]
    b_v = None if b is None else b[2*E:]
    
    with torch.no_grad():
        # Project Q from last token only
        q = F.linear(last, W_q, b_q)  # [1, B, E]
        
        # Project K,V from full sequence
        k_full = F.linear(full, W_k, b_k)  # [T, B, E]
        v_full = F.linear(full, W_v, b_v)  # [T, B, E]
        
        # Simulate KV cache: prefix (0..T-2) + last token
        k_prev = k_full[:-1]  # [T-1, B, E] - cached from previous steps
        v_prev = v_full[:-1]  # [T-1, B, E]
        k_last = k_full[-1:]  # [1, B, E] - new computation
        v_last = v_full[-1:]  # [1, B, E]
        
        # Split heads helper
        def split_heads(t):
            """[L, B, E] -> [B, H, L, Dh]"""
            L, B_inner, _ = t.shape
            return t.view(L, B_inner, heads, Dh).permute(1, 2, 0, 3).contiguous()
        
        # Split heads
        qh = split_heads(q)  # [B, H, 1, Dh]
        
        # Concatenate cached K,V with new K,V
        Kh = torch.cat([split_heads(k_prev), split_heads(k_last)], dim=2)  # [B, H, T, Dh]
        Vh = torch.cat([split_heads(v_prev), split_heads(v_last)], dim=2)  # [B, H, T, Dh]
        
        # Scaled dot-product attention
        scores = (qh @ Kh.transpose(-2, -1)) / (Dh ** 0.5)  # [B, H, 1, T]
        attn = scores.softmax(dim=-1)                       # [B, H, 1, T]
        ctx = attn @ Vh                                     # [B, H, 1, Dh]
        
        # Merge heads: [B, H, 1, Dh] -> [1, B, E]
        ctx = ctx.permute(2, 0, 1, 3).contiguous().view(1, B, E)
        
        # Apply output projection (critical!)
        y_inc = mha.out_proj(ctx)  # [1, B, E]
    
    # Compare results
    max_err = (y_stock - y_inc).abs().max().item()
    return max_err


def run_tests():
    """Run multiple tests with different configurations."""
    print("Testing KV cache math correctness...\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test configurations
    configs = [
        {"E": 512, "heads": 8, "T": 5, "B": 2},   # Small test
        {"E": 512, "heads": 8, "T": 20, "B": 4},  # Medium test
        {"E": 512, "heads": 8, "T": 50, "B": 8},  # Larger test
        {"E": 256, "heads": 4, "T": 10, "B": 3},  # Different dims
    ]
    
    all_passed = True
    
    for i, cfg in enumerate(configs, 1):
        print(f"Test {i}: E={cfg['E']}, heads={cfg['heads']}, T={cfg['T']}, B={cfg['B']}")
        
        # Create MHA module
        mha = nn.MultiheadAttention(
            embed_dim=cfg['E'],
            num_heads=cfg['heads'],
            dropout=0.0,
            batch_first=False
        )
        
        # Run test 3 times with different random seeds
        max_errors = []
        for seed in range(3):
            torch.manual_seed(seed)
            err = check_sa_incremental_equiv(
                mha, 
                T=cfg['T'], 
                B=cfg['B'], 
                E=cfg['E'], 
                heads=cfg['heads'],
                device=device
            )
            max_errors.append(err)
        
        avg_err = sum(max_errors) / len(max_errors)
        max_err = max(max_errors)
        
        print(f"  Max error: {max_err:.2e}, Avg error: {avg_err:.2e}")
        
        if max_err > 1e-5:
            print(f"  ❌ FAILED! Error too large")
            all_passed = False
        else:
            print(f"  ✅ PASSED")
        print()
    
    if all_passed:
        print("🎉 All tests PASSED! KV cache math is correct.")
        print("\nThe incremental KV cache produces identical results to full attention.")
        print("Safe to proceed with Step 3: implementing in the actual model.")
    else:
        print("⚠️ Some tests FAILED. Do not proceed until fixed.")
    
    return all_passed


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)