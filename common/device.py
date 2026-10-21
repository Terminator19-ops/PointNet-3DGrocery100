"""Device selection helpers shared by training and evaluation scripts."""

from __future__ import annotations

import torch


def xpu_is_available() -> bool:
    """Return whether this PyTorch build can see an Intel XPU device."""
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return False
    try:
        return bool(xpu.is_available())
    except RuntimeError:
        return False


def choose_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto``, CPU, CUDA, or Intel XPU to a usable device."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if xpu_is_available():
            return torch.device("xpu")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if requested == "xpu" and not xpu_is_available():
        raise RuntimeError(
            "XPU was requested but no Intel GPU is available to PyTorch. "
            "Install the XPU wheel and the system Level Zero runtime, then verify "
            "that /dev/dri/renderD128 is accessible."
        )
    return torch.device(requested)


def seed_device(seed: int) -> None:
    """Seed accelerator-specific random number generators when available."""
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu_is_available():
        xpu.manual_seed_all(seed)