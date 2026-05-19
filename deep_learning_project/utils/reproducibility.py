"""Reproducibility helpers: seed setting and device selection."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Also disables cuDNN non-determinism so that results are reproducible
    across runs on the same hardware.

    Parameters
    ----------
    seed:
        Integer seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return the best available compute device.

    Priority order: CUDA > MPS (Apple Silicon) > CPU.
    Prints a message indicating which device was selected.

    Returns
    -------
    torch.device
        The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")
    return device
