"""Utilities for computing, storing, and loading Meta-World vector normalization stats."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


STATS_GROUP_NAME = "stats"
MIN_STD = 1e-6


def _normalize_stats_name(column: str) -> tuple[str, str]:
    return f"{column}_mean", f"{column}_std"


def init_running_vector_stats(dim: int) -> dict[str, np.ndarray | int]:
    return {
        "count": 0,
        "sum": np.zeros((dim,), dtype=np.float64),
        "sum_sq": np.zeros((dim,), dtype=np.float64),
    }


def update_running_vector_stats(
    stats: dict[str, np.ndarray | int],
    values: np.ndarray,
) -> None:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]

    if values.ndim != 2:
        raise ValueError(
            "Expected vector values with shape (N, D) or (N,), "
            f"got shape {tuple(values.shape)}."
        )

    stats["count"] += int(values.shape[0])
    stats["sum"] += values.sum(axis=0)
    stats["sum_sq"] += np.square(values).sum(axis=0)


def finalize_running_vector_stats(
    stats: dict[str, np.ndarray | int],
) -> tuple[np.ndarray, np.ndarray]:
    count = int(stats["count"])
    if count <= 0:
        raise ValueError("Cannot finalize vector stats with zero samples.")

    mean = stats["sum"] / count
    var = (stats["sum_sq"] / count) - np.square(mean)
    var = np.maximum(var, MIN_STD**2)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def write_vector_stats(
    h5_file: h5py.File,
    column: str,
    mean: np.ndarray,
    std: np.ndarray,
) -> None:
    stats_group = h5_file.require_group(STATS_GROUP_NAME)
    mean_name, std_name = _normalize_stats_name(column)

    if mean_name in stats_group:
        del stats_group[mean_name]
    if std_name in stats_group:
        del stats_group[std_name]

    stats_group.create_dataset(mean_name, data=np.asarray(mean, dtype=np.float32))
    stats_group.create_dataset(std_name, data=np.asarray(std, dtype=np.float32))


def load_vector_stats(
    h5_path: str | Path,
    column: str,
):
    import torch

    h5_path = Path(h5_path)
    mean_name, std_name = _normalize_stats_name(column)

    with h5py.File(h5_path, "r") as h5_file:
        if STATS_GROUP_NAME not in h5_file:
            raise KeyError(
                f"Missing '{STATS_GROUP_NAME}/' group in dataset {h5_path}."
            )

        stats_group = h5_file[STATS_GROUP_NAME]
        if mean_name not in stats_group or std_name not in stats_group:
            raise KeyError(
                f"Missing saved stats for '{column}' in dataset {h5_path}. "
                f"Expected '{STATS_GROUP_NAME}/{mean_name}' and "
                f"'{STATS_GROUP_NAME}/{std_name}'."
            )

        mean = torch.from_numpy(np.asarray(stats_group[mean_name][()], dtype=np.float32))
        std = torch.from_numpy(np.asarray(stats_group[std_name][()], dtype=np.float32))

    return mean, std.clamp_min(MIN_STD)


def normalize_vector_with_stats(
    values,
    mean,
    std,
):
    import torch

    values = torch.as_tensor(values).float()
    mean = mean.to(device=values.device, dtype=values.dtype)
    std = std.to(device=values.device, dtype=values.dtype).clamp_min(MIN_STD)
    return (values - mean) / std


def get_saved_vector_normalizer(dataset, source: str, target: str):
    from stable_pretraining import data as dt

    mean, std = load_vector_stats(dataset.h5_path, source)

    def norm_fn(x):
        return normalize_vector_with_stats(x, mean, std)

    return dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
