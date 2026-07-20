from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .config import MANUAL_DATASET_PATH


def build_states(raw_obs: list[dict]) -> np.ndarray:
    """Concatenate observation and desired goal into a flat state vector."""
    return np.asarray(
        [np.concatenate([sample["observation"], sample["desired_goal"]]) for sample in raw_obs],
        dtype=np.float32,
    )


def load_manual_dataset(dataset_path: Path | str = MANUAL_DATASET_PATH) -> tuple[np.ndarray, np.ndarray]:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with dataset_path.open("rb") as file:
        data = pickle.load(file)

    raw_obs = data["obs"]
    actions = np.asarray(data["actions"], dtype=np.float32)
    states = build_states(raw_obs)
    return states, actions
