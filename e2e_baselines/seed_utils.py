#!/usr/bin/env python3
"""
Add proper seeding for reproducible training
"""
import torch
import numpy as np
import random
import os

def seed_everything(seed=42):
    """
    Seed all random number generators for reproducibility
    
    Args:
        seed: Random seed (default: 42)
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variable for PyTorch
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ Seeded all RNGs with seed={seed}")
    print(f"  - Python random: {seed}")
    print(f"  - Numpy: {seed}")
    print(f"  - PyTorch: {seed}")
    print(f"  - CUDA: {seed}")
    print(f"  - CuDNN deterministic: True")
    print(f"  - CuDNN benchmark: False")

if __name__ == "__main__":
    seed_everything(42)
    print("\n✓ All random number generators seeded!")
