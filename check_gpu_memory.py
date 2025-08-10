#!/usr/bin/env python3
"""
Utility script to check GPU memory usage and clean up if needed.
"""

import torch
import subprocess
import sys


def get_gpu_memory_info():
    """Get GPU memory usage information."""
    if torch.cuda.is_available():
        print("🎮 CUDA GPU Information:")
        print("-" * 40)
        
        # PyTorch info
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\nDevice {i}: {props.name}")
            print(f"  Total memory: {props.total_memory / 1024**3:.2f} GB")
            
            # Current usage
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Cached: {cached:.2f} GB")
            print(f"  Free: {(props.total_memory / 1024**3) - cached:.2f} GB")
        
        # Try nvidia-smi for more details
        try:
            print("\n" + "=" * 40)
            print("nvidia-smi output:")
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print(result.stdout)
        except FileNotFoundError:
            print("nvidia-smi not found")
            
    elif torch.backends.mps.is_available():
        print("🍎 MPS (Metal) GPU Information:")
        print("-" * 40)
        print("MPS is available")
        print("Note: MPS doesn't provide detailed memory stats")
        
        # Check if any tensors are on MPS
        try:
            test = torch.zeros(1).to('mps')
            print("✓ Can create tensors on MPS")
            del test
        except Exception as e:
            print(f"✗ Error creating MPS tensor: {e}")
            
    else:
        print("❌ No GPU detected (CUDA or MPS)")
        print("Running on CPU only")


def clear_gpu_memory():
    """Force clear GPU memory."""
    print("\n🧹 Clearing GPU memory...")
    
    if torch.cuda.is_available():
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✓ CUDA cache cleared")
        
        # Show memory after clearing
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  Device {i} - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
            
    elif torch.backends.mps.is_available():
        if hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        print("✓ MPS synchronized")
    else:
        print("No GPU to clear")


def main():
    """Main function."""
    print("=" * 50)
    print("GPU MEMORY CHECKER")
    print("=" * 50)
    
    # Check current status
    get_gpu_memory_info()
    
    # Ask if user wants to clear
    if len(sys.argv) > 1 and sys.argv[1] == '--clear':
        clear_gpu_memory()
        print("\n" + "=" * 50)
        print("After cleanup:")
        get_gpu_memory_info()


if __name__ == "__main__":
    main()