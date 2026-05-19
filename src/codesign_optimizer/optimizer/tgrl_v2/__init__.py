from __future__ import annotations


def ensure_torch_available() -> None:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TG-RL v2 requires PyTorch. Install it with: pip install -e \".[dev,rl]\""
        ) from exc
