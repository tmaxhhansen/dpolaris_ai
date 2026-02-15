"""
Runtime device detection for deep-learning workloads.
"""

from __future__ import annotations

import os
from typing import Any, Optional


VALID_DEVICE_PREFERENCES = {"auto", "cuda", "mps", "cpu"}


def _normalize_preference(preference: Optional[str]) -> tuple[str, Optional[str]]:
    raw = (preference or os.getenv("DPOLARIS_DEVICE", "auto")).strip().lower()
    if raw in VALID_DEVICE_PREFERENCES:
        return raw, None
    return "auto", f"Invalid DPOLARIS_DEVICE='{raw}', falling back to auto."


def detect_device(preference: Optional[str] = None) -> dict[str, Any]:
    """
    Detect the runtime compute device for PyTorch workloads.

    Preference order:
    - Explicit preference via argument or DPOLARIS_DEVICE env (auto|cuda|mps|cpu)
    - auto => cuda, then mps, then cpu
    """
    requested, warning = _normalize_preference(preference)
    info: dict[str, Any] = {
        "requested": requested,
        "device": "cpu",
        "reason": "CPU fallback",
        "torch_version": None,
        "cuda_available": False,
        "mps_available": False,
        "gpu_name": None,
        "warning": warning,
        "torch_importable": False,
    }

    try:
        import torch  # type: ignore
    except Exception as exc:
        info["reason"] = f"PyTorch import failed: {exc}"
        return info

    info["torch_importable"] = True
    info["torch_version"] = getattr(torch, "__version__", None)

    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    info["cuda_available"] = cuda_available
    info["mps_available"] = mps_available
    info["gpu_name"] = gpu_name

    if requested == "cpu":
        info["device"] = "cpu"
        info["reason"] = "DPOLARIS_DEVICE=cpu requested."
        return info

    if requested == "cuda":
        if cuda_available:
            info["device"] = "cuda"
            info["reason"] = "DPOLARIS_DEVICE=cuda requested and CUDA is available."
        else:
            info["device"] = "cpu"
            info["reason"] = "DPOLARIS_DEVICE=cuda requested but CUDA is unavailable; falling back to CPU."
        return info

    if requested == "mps":
        if mps_available:
            info["device"] = "mps"
            info["reason"] = "DPOLARIS_DEVICE=mps requested and MPS is available."
        else:
            info["device"] = "cpu"
            info["reason"] = "DPOLARIS_DEVICE=mps requested but MPS is unavailable; falling back to CPU."
        return info

    if cuda_available:
        info["device"] = "cuda"
        info["reason"] = "Auto-selected CUDA."
        return info

    if mps_available:
        info["device"] = "mps"
        info["reason"] = "Auto-selected Apple MPS."
        return info

    info["device"] = "cpu"
    info["reason"] = "Auto-selected CPU fallback."
    return info

