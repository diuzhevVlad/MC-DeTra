"""Checkpoint helpers shared by training, evaluation, and visualization."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint while preserving compatibility with older torch builds."""

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_compatible(module: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Load only matching tensor keys into ``module`` and return skipped keys."""

    own = module.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in own and own[key].shape == value.shape
    }
    skipped = [key for key in state_dict if key not in compatible]
    module.load_state_dict(compatible, strict=False)
    return skipped


def capture_rng_state() -> dict[str, object]:
    """Capture process RNG state for full-state training resume."""

    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object] | None) -> None:
    """Restore RNG state from a full-state checkpoint when available."""

    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically write a torch checkpoint payload."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
