#!/usr/bin/env python3
"""Test MPS performance vs CPU for table structure model"""

import os
import torch
import time
from mps_diagnostics import check_mps_setup, benchmark_attention

# Disable MPS fallback to catch issues
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

def main():
    print("="*60)
    print("MPS vs CPU Performance Comparison")
    print("="*60)
    
    # Check MPS availability
    if not torch.backends.mps.is_available():
        print("❌ MPS not available on this system")
        return
    
    check_mps_setup()
    
    # Benchmark on CPU
    print("\n" + "="*60)
    print("CPU BENCHMARKS")
    print("="*60)
    benchmark_attention(device='cpu', use_sdpa=False)
    print("\nCPU with SDPA:")
    benchmark_attention(device='cpu', use_sdpa=True)
    
    # Benchmark on MPS
    print("\n" + "="*60)
    print("MPS BENCHMARKS")
    print("="*60)
    benchmark_attention(device='mps', use_sdpa=False)
    print("\nMPS with SDPA:")
    benchmark_attention(device='mps', use_sdpa=True)
    
    # Test the actual workload size from your model
    print("\n" + "="*60)
    print("REALISTIC WORKLOAD TEST")
    print("="*60)
    
    from transformer_optimized import TMTransformerDecoderLayer
    
    # Your actual sizes during inference
    T = 1  # Last token only
    S_encoder = 28 * 28  # 784 - encoder output positions
    B = 1
    D = 512
    
    print(f"Testing with your actual sizes: T={T}, S={S_encoder}, B={B}, D={D}")
    print(f"Work size (T*S) = {T * S_encoder}")
    
    for device in ['cpu', 'mps']:
        for use_sdpa in [False, True]:
            layer = TMTransformerDecoderLayer(
                d_model=D, nhead=8, dim_feedforward=1024, use_sdpa=use_sdpa
            ).eval().to(device)
            
            q = torch.randn(T, B, D, device=device)
            mem = torch.randn(S_encoder, B, D, device=device)
            
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    layer(q, mem)
            
            if device == 'mps':
                torch.mps.synchronize()
            
            # Time
            t0 = time.time()
            for _ in range(100):
                with torch.no_grad():
                    layer(q, mem)
            
            if device == 'mps':
                torch.mps.synchronize()
            
            time_ms = (time.time() - t0) / 100 * 1000
            print(f"  {device:3s} SDPA={use_sdpa}: {time_ms:6.2f} ms/step")

if __name__ == "__main__":
    main()