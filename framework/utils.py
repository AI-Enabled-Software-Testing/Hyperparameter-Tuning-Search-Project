import torch
from typing import Tuple
from torch.nn import Module


def get_device() -> torch.device:
    """Return the preferred torch.device."""
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


def is_cuda_available() -> bool:
    """Return True when CUDA is accessible."""
    return torch.cuda.is_available()


def count_parameters(model: Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model."""
    total_params = sum(torch.numel(p) for p in model.parameters())
    trainable_params = sum(
        torch.numel(p) for p in model.parameters() if p.requires_grad
    )
    return total_params, trainable_params


def torch_version() -> str:
    """Get the installed PyTorch version."""
    return torch.__version__


def test_pytorch_setup() -> None:
    """Quickly report the local PyTorch environment."""
    device = get_device()
    print(f"PyTorch version: {torch_version()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {device.type}")
    if device.type == "cuda":
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(device)}")
    print("PyTorch setup test completed.")
