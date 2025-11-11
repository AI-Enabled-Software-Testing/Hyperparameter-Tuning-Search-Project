import torch
from typing import Optional, Tuple
from torch.nn import Module

_device: torch.device = torch.device("cpu")
_is_cuda: bool = False


def init_device(device_str: Optional[str] = None) -> None:
    global _device, _is_cuda
    if device_str:
        _device = torch.device(device_str)
    else:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _is_cuda = _device.type == "cuda"


def device() -> torch.device:
    """Get the current device."""
    return _device


def is_cuda() -> bool:
    """Check if the current device is CUDA."""
    return _is_cuda


def count_parameters(model: Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model."""
    total_params = sum(torch.numel(p) for p in model.parameters())
    trainable_params = sum(torch.numel(p) for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

