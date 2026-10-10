"""Device selection and GPU memory helpers."""
from __future__ import annotations


def select_device(requested: str | None = "auto") -> str:
    """Return 'cpu' or 'cuda:N'. 'auto' prefers CUDA when available; an explicit CUDA request without CUDA raises."""
    import torch

    req = (requested or "auto").lower()
    if req == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if req == "cpu":
        return "cpu"
    if req.startswith("cuda") or req.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError(f"Device '{requested}' requested but CUDA is not available")
        return req if req.startswith("cuda") else f"cuda:{req}"
    raise ValueError(f"Unknown device '{requested}' (use auto, cpu, or cuda:N)")


def _idx(device: str) -> int:
    import torch

    torch.cuda.init()    # memory-stat C APIs fail with "Invalid device argument" before CUDA is initialised
    return int(device.split(":")[1]) if ":" in device else 0


def reset_peak_memory(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(_idx(device))


def peak_gpu_memory_mb(device: str) -> float | None:
    """Peak allocated GPU memory (MiB) since the last reset; None on CPU."""
    import torch

    if not device.startswith("cuda"):
        return None
    return torch.cuda.max_memory_allocated(_idx(device)) / 2**20


def sync(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize(_idx(device))


def free_gpu() -> None:
    """Release cached GPU memory so models can run one after another."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
