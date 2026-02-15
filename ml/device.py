"""
Device selection utility for dPolaris deep-learning workloads.

Usage:
    from ml.device import get_device_info, get_torch_device

    # Get comprehensive device info
    info = get_device_info()
    print(info["device"])  # "cuda", "mps", or "cpu"
    print(info["reason"])  # Why this device was selected

    # Get torch.device directly
    device = get_torch_device()  # Returns torch.device or raises ImportError

Environment Variables:
    DPOLARIS_DEVICE: auto|cuda|mps|cpu (default: auto)

Selection Logic (auto mode):
    1. CUDA if available (Windows/Linux GPU)
    2. MPS if available (Apple Silicon)
    3. CPU fallback
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger("dpolaris.ml.device")

VALID_DEVICE_PREFERENCES = {"auto", "cuda", "mps", "cpu"}


def _normalize_preference(preference: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Normalize device preference from argument or environment."""
    raw = (preference or os.getenv("DPOLARIS_DEVICE", "auto")).strip().lower()
    if raw in VALID_DEVICE_PREFERENCES:
        return raw, None
    return "auto", f"Invalid DPOLARIS_DEVICE='{raw}', falling back to auto."


def get_device_info(preference: Optional[str] = None) -> dict[str, Any]:
    """
    Detect the runtime compute device for PyTorch workloads.

    Args:
        preference: Override for DPOLARIS_DEVICE (auto|cuda|mps|cpu)

    Returns:
        Dictionary with:
        - device: "cuda", "mps", or "cpu"
        - reason: Human-readable explanation
        - torch_version: PyTorch version string or None
        - cuda_available: bool
        - mps_available: bool
        - gpu_name: CUDA device name or None
        - torch_importable: bool
        - warning: Optional warning message
    """
    requested, warning = _normalize_preference(preference)

    info: dict[str, Any] = {
        "requested": requested,
        "device": "cpu",
        "reason": "CPU fallback (torch not available)",
        "torch_version": None,
        "cuda_available": False,
        "mps_available": False,
        "gpu_name": None,
        "warning": warning,
        "torch_importable": False,
    }

    try:
        import torch  # type: ignore
    except ImportError as exc:
        info["reason"] = f"PyTorch not installed: {exc}"
        logger.warning("PyTorch not available: %s", exc)
        return info
    except Exception as exc:
        info["reason"] = f"PyTorch import failed: {exc}"
        logger.warning("PyTorch import error: %s", exc)
        return info

    info["torch_importable"] = True
    info["torch_version"] = getattr(torch, "__version__", "unknown")

    # Detect available devices
    cuda_available = False
    mps_available = False
    gpu_name = None

    try:
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception as exc:
        logger.debug("CUDA detection failed: %s", exc)

    try:
        mps_available = bool(
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    except Exception as exc:
        logger.debug("MPS detection failed: %s", exc)

    info["cuda_available"] = cuda_available
    info["mps_available"] = mps_available
    info["gpu_name"] = gpu_name

    # Select device based on preference
    if requested == "cpu":
        info["device"] = "cpu"
        info["reason"] = "DPOLARIS_DEVICE=cpu explicitly requested."
        logger.info("Using CPU (explicitly requested)")
        return info

    if requested == "cuda":
        if cuda_available:
            info["device"] = "cuda"
            info["reason"] = f"DPOLARIS_DEVICE=cuda requested. GPU: {gpu_name}"
            logger.info("Using CUDA: %s", gpu_name)
        else:
            info["device"] = "cpu"
            info["reason"] = "DPOLARIS_DEVICE=cuda requested but CUDA unavailable; CPU fallback."
            info["warning"] = "CUDA requested but not available"
            logger.warning("CUDA requested but unavailable, falling back to CPU")
        return info

    if requested == "mps":
        if mps_available:
            info["device"] = "mps"
            info["reason"] = "DPOLARIS_DEVICE=mps requested and Apple MPS is available."
            logger.info("Using Apple MPS")
        else:
            info["device"] = "cpu"
            info["reason"] = "DPOLARIS_DEVICE=mps requested but MPS unavailable; CPU fallback."
            info["warning"] = "MPS requested but not available"
            logger.warning("MPS requested but unavailable, falling back to CPU")
        return info

    # Auto mode: prioritize CUDA > MPS > CPU
    if cuda_available:
        info["device"] = "cuda"
        info["reason"] = f"Auto-selected CUDA. GPU: {gpu_name}"
        logger.info("Auto-selected CUDA: %s", gpu_name)
        return info

    if mps_available:
        info["device"] = "mps"
        info["reason"] = "Auto-selected Apple MPS (Metal Performance Shaders)."
        logger.info("Auto-selected Apple MPS")
        return info

    info["device"] = "cpu"
    info["reason"] = "Auto-selected CPU (no GPU acceleration available)."
    logger.info("Using CPU (no GPU available)")
    return info


def get_torch_device(preference: Optional[str] = None):
    """
    Get a torch.device object for the selected device.

    Args:
        preference: Override for DPOLARIS_DEVICE

    Returns:
        torch.device instance

    Raises:
        ImportError: If PyTorch is not installed
    """
    import torch  # Will raise ImportError if not installed

    info = get_device_info(preference)
    return torch.device(info["device"])


def is_torch_available() -> bool:
    """Check if PyTorch can be imported."""
    try:
        import torch  # type: ignore # noqa: F401
        return True
    except Exception:
        return False


def is_sklearn_available() -> bool:
    """Check if scikit-learn can be imported."""
    try:
        from sklearn.preprocessing import StandardScaler  # noqa: F401
        return True
    except Exception:
        return False


def get_dependency_status() -> dict[str, Any]:
    """
    Get status of ML dependencies.

    Returns:
        Dictionary with:
        - torch_available: bool
        - torch_error: Optional error message
        - sklearn_available: bool
        - sklearn_error: Optional error message
        - deep_learning_ready: bool (both torch and sklearn available)
    """
    result: dict[str, Any] = {
        "torch_available": False,
        "torch_error": None,
        "sklearn_available": False,
        "sklearn_error": None,
        "deep_learning_ready": False,
    }

    try:
        import torch  # type: ignore # noqa: F401
        result["torch_available"] = True
    except Exception as exc:
        result["torch_error"] = str(exc)

    try:
        from sklearn.preprocessing import StandardScaler  # noqa: F401
        result["sklearn_available"] = True
    except Exception as exc:
        result["sklearn_error"] = str(exc)

    result["deep_learning_ready"] = (
        result["torch_available"] and result["sklearn_available"]
    )

    return result


# Module-level convenience for logging device on import
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    info = get_device_info()
    print(json.dumps(info, indent=2))
