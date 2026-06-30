"""
Device Utilities - Auto-detect GPU/CPU for model inference
Automatically selects GPU if available, otherwise falls back to CPU.
Supports CUDA (NVIDIA) and MPS (Apple Silicon).
"""
import torch


def get_device() -> str:
    """
    Auto-detect the best available device for model inference.
    
    Returns:
        str: 'cuda:0' if CUDA GPU is available,
             'mps' if Apple Silicon GPU is available,
             'cpu' otherwise (fallback)
    """
    if torch.cuda.is_available():
        device = 'cuda:0'
        print(f"[Device] GPU detected: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("[Device] Apple MPS GPU detected")
    else:
        device = 'cpu'
        print("[Device] No GPU detected, using CPU")
    return device