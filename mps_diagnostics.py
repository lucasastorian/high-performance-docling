import torch
import time
import os

def check_mps_setup():
    """Comprehensive MPS diagnostics"""
    print("\n" + "="*60)
    print("🔍 MPS DIAGNOSTICS")
    print("="*60)
    
    # Check MPS availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # Check environment
    fallback = os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'not set')
    print(f"PYTORCH_ENABLE_MPS_FALLBACK: {fallback}")
    
    if torch.backends.mps.is_available():
        # Test basic MPS operation
        try:
            test_tensor = torch.randn(10, 10, device='mps')
            result = torch.matmul(test_tensor, test_tensor)
            torch.mps.synchronize()
            print("✅ MPS basic operations working")
        except Exception as e:
            print(f"❌ MPS basic test failed: {e}")
    
    print("="*60 + "\n")
    return torch.backends.mps.is_available()

def assert_on_device(device, *tensors, name=""):
    """Assert all tensors are on the expected device"""
    dev_str = device if isinstance(device, str) else device.type
    for i, t in enumerate(tensors):
        if t is not None and isinstance(t, torch.Tensor):
            if t.device.type != dev_str:
                raise AssertionError(
                    f"{name} Tensor {i} on {t.device}, expected {dev_str}"
                )

def check_model_device(model, expected_device='mps'):
    """Check if entire model is on expected device"""
    dev_str = expected_device if isinstance(expected_device, str) else expected_device.type
    wrong_params = []
    for name, param in model.named_parameters():
        if param.device.type != dev_str:
            wrong_params.append((name, param.device))
    
    if wrong_params:
        print(f"⚠️ Found {len(wrong_params)} parameters not on {dev_str}:")
        for name, device in wrong_params[:5]:  # Show first 5
            print(f"  - {name}: {device}")
        return False
    else:
        print(f"✅ All model parameters on {dev_str}")
        return True

def timed_with_sync(fn, device, reps=10, warmup=3):
    """Time a function with proper device synchronization"""
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Synchronize before timing
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()
    
    t0 = time.time()
    for _ in range(reps):
        fn()
    
    # Synchronize after timing
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()
    
    return (time.time() - t0) / reps

def benchmark_attention(device='mps', use_sdpa=True):
    """Benchmark attention on different devices"""
    from transformer_optimized import TMTransformerDecoderLayer
    
    print(f"\n📊 Benchmarking attention on {device} (SDPA={use_sdpa})")
    print("-" * 40)
    
    # Test different sizes
    configs = [
        # (name, T, S, B, D)
        ("Last-token (your case)", 1, 128, 1, 512),
        ("Small sequence", 16, 128, 1, 512),
        ("Medium sequence", 64, 256, 1, 512),
        ("Large sequence", 128, 512, 1, 512),
    ]
    
    for name, T, S, B, D in configs:
        layer = TMTransformerDecoderLayer(
            d_model=D, nhead=8, dim_feedforward=1024, use_sdpa=use_sdpa
        ).eval().to(device)
        
        q = torch.randn(T, B, D, device=device)
        mem = torch.randn(S, B, D, device=device)
        
        def run():
            with torch.no_grad():
                return layer(q, mem)
        
        time_ms = timed_with_sync(run, device, reps=20) * 1000
        print(f"  {name:30s} T={T:3d} S={S:3d}: {time_ms:6.2f} ms")

def should_use_sdpa(device, query_len, key_len):
    """Heuristic for when to use SDPA"""
    work = query_len * key_len
    
    if device in ('cuda', 'mps'):
        # Use SDPA for larger workloads on GPU
        return work >= 256
    else:
        # Higher threshold for CPU
        return work >= 512