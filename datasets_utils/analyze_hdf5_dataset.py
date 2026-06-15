"""Streaming sanity checks for converted or raw Meta-World HDF5 datasets.

The script is intentionally conservative: it does not load the full dataset into
RAM, and it focuses on signals that usually explain strange model rollouts:
non-finite values, suspicious ranges, action saturation, blank/constant image
frames, episode/success statistics, and large per-step jumps in vector signals.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass


SKIP_KEYS = {"ep_len", "ep_offset"}
STATS_GROUP_NAME = "stats"
IMAGE_LIKE_KEYS = {"pixels", "depth", "tactile"}
VECTOR_JUMP_KEYS = (
    "action",
    "proprio",
    "force_torque",
    "ee_position",
    "ee_xyz",
    "object_1_xyz",
    "object_2_xyz",
    "target_pos",
    "qpos",
    "qvel",
)
EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze an HDF5 dataset for NaNs/Infs, suspicious ranges, outliers, "
            "blank images, episode statistics, and large trajectory jumps."
        )
    )
    parser.add_argument(
        "dataset",
        type=str,
        help=(
            "Dataset path or dataset name. If a name is passed, the script looks "
            "for <name>.h5 or <name>.hdf5 under --cache-dir."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory used when dataset is given by name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write report files. Defaults next to the dataset.",
    )
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--sample-values", type=int, default=200_000)
    parser.add_argument("--sample-row-norms", type=int, default=100_000)
    parser.add_argument(
        "--max-jump-episodes",
        type=int,
        default=2000,
        help="Maximum number of episodes used for per-step jump checks.",
    )
    parser.add_argument("--max-jump-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_dataset_path(dataset: str, cache_dir: Path | None) -> Path:
    path = Path(dataset).expanduser()
    if path.exists():
        return path

    root = cache_dir or Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable_worldmodel"))
    root = root.expanduser()
    for suffix in (".h5", ".hdf5"):
        candidate = root / f"{dataset}{suffix}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find dataset {dataset!r}. Tried direct path and {root}/<name>.h5/.hdf5."
    )


def is_numeric_dtype(dtype) -> bool:
    return np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_)


def append_sample(existing: list[np.ndarray], values: np.ndarray, limit: int, rng) -> None:
    if limit <= 0 or values.size == 0:
        return
    values = np.asarray(values).reshape(-1)
    if values.size > limit:
        values = values[rng.choice(values.size, size=limit, replace=False)]
    existing.append(values.astype(np.float64, copy=False))

    total = sum(chunk.size for chunk in existing)
    if total <= limit * 2:
        return

    merged = np.concatenate(existing)
    if merged.size > limit:
        merged = merged[rng.choice(merged.size, size=limit, replace=False)]
    existing.clear()
    existing.append(merged)


def finalize_sample(chunks: list[np.ndarray], limit: int, rng) -> np.ndarray:
    if not chunks:
        return np.asarray([], dtype=np.float64)
    values = np.concatenate(chunks).astype(np.float64, copy=False)
    if values.size > limit:
        values = values[rng.choice(values.size, size=limit, replace=False)]
    return values


def percentiles(values: np.ndarray) -> dict[str, float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    qs = [0, 0.1, 1, 5, 25, 50, 75, 95, 99, 99.9, 100]
    vals = np.percentile(values, qs)
    return {f"p{q:g}": float(v) for q, v in zip(qs, vals)}


def sample_iqr_outlier_fraction(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 8:
        return None
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr <= EPS:
        return 0.0
    lo = q1 - 3.0 * iqr
    hi = q3 + 3.0 * iqr
    return float(np.mean((values < lo) | (values > hi)))


@dataclass
class ColumnAnalyzer:
    name: str
    dtype: str
    shape: tuple[int, ...]
    max_sample_values: int
    max_sample_row_norms: int
    rng: np.random.Generator
    row_count: int = 0
    element_count: int = 0
    finite_count: int = 0
    nan_count: int = 0
    posinf_count: int = 0
    neginf_count: int = 0
    bad_row_count: int = 0
    min_value: float | None = None
    max_value: float | None = None
    sum_value: float = 0.0
    sumsq_value: float = 0.0
    sample_chunks: list[np.ndarray] = field(default_factory=list)
    row_norm_chunks: list[np.ndarray] = field(default_factory=list)
    image_row_count: int = 0
    low_variance_rows: int = 0
    blackish_rows: int = 0
    action_saturated_elements: int = 0
    action_out_of_range_elements: int = 0

    def update(self, data: np.ndarray) -> None:
        if data.shape[0] == 0:
            return

        self.row_count += int(data.shape[0])
        self.element_count += int(data.size)

        floating = np.issubdtype(data.dtype, np.floating)
        if floating:
            finite_mask = np.isfinite(data)
            finite_values = data[finite_mask]
            self.nan_count += int(np.isnan(data).sum())
            self.posinf_count += int(np.isposinf(data).sum())
            self.neginf_count += int(np.isneginf(data).sum())
            row_finite = finite_mask.reshape(data.shape[0], -1).all(axis=1)
            self.bad_row_count += int((~row_finite).sum())
        else:
            finite_values = data.reshape(-1)

        self.finite_count += int(finite_values.size)
        if finite_values.size:
            values64 = finite_values.astype(np.float64, copy=False)
            chunk_min = float(np.min(values64))
            chunk_max = float(np.max(values64))
            self.min_value = (
                chunk_min if self.min_value is None else min(self.min_value, chunk_min)
            )
            self.max_value = (
                chunk_max if self.max_value is None else max(self.max_value, chunk_max)
            )
            self.sum_value += float(np.sum(values64))
            self.sumsq_value += float(np.sum(values64 * values64))
            append_sample(
                self.sample_chunks,
                values64,
                self.max_sample_values,
                self.rng,
            )

        flat_rows = data.reshape(data.shape[0], -1)
        if flat_rows.shape[1] <= 128:
            if floating:
                good_rows = np.isfinite(flat_rows).all(axis=1)
                row_values = flat_rows[good_rows].astype(np.float64, copy=False)
            else:
                row_values = flat_rows.astype(np.float64, copy=False)
            if row_values.size:
                norms = np.linalg.norm(row_values, axis=1)
                append_sample(
                    self.row_norm_chunks,
                    norms,
                    self.max_sample_row_norms,
                    self.rng,
                )

        if self.name in IMAGE_LIKE_KEYS or data.ndim >= 3:
            self._update_image_like(data, floating=floating)

        if self.name == "action":
            values = finite_values.astype(np.float64, copy=False)
            self.action_saturated_elements += int(np.sum(np.abs(values) >= 0.999))
            self.action_out_of_range_elements += int(np.sum(np.abs(values) > 1.0001))

    def _update_image_like(self, data: np.ndarray, *, floating: bool) -> None:
        flat_rows = data.reshape(data.shape[0], -1)
        if floating:
            finite_rows = np.isfinite(flat_rows).all(axis=1)
            flat_rows = flat_rows[finite_rows]
        if flat_rows.size == 0:
            return

        values = flat_rows.astype(np.float32, copy=False)
        row_means = values.mean(axis=1)
        row_stds = values.std(axis=1)
        self.image_row_count += int(values.shape[0])
        self.low_variance_rows += int(np.sum(row_stds <= 1e-6))
        if np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.bool_):
            self.blackish_rows += int(np.sum((row_means <= 2.0) & (row_stds <= 1.0)))
        else:
            self.blackish_rows += int(np.sum((row_means <= 1e-6) & (row_stds <= 1e-6)))

    def summarize(self) -> dict[str, Any]:
        sample = finalize_sample(self.sample_chunks, self.max_sample_values, self.rng)
        row_norms = finalize_sample(
            self.row_norm_chunks,
            self.max_sample_row_norms,
            self.rng,
        )
        mean = self.sum_value / self.finite_count if self.finite_count else None
        var = None
        std = None
        if self.finite_count:
            var = max(self.sumsq_value / self.finite_count - float(mean) ** 2, 0.0)
            std = float(np.sqrt(var))

        summary = {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "rows": int(self.row_count),
            "elements": int(self.element_count),
            "finite_ratio": (
                float(self.finite_count / self.element_count)
                if self.element_count
                else None
            ),
            "nan_count": int(self.nan_count),
            "posinf_count": int(self.posinf_count),
            "neginf_count": int(self.neginf_count),
            "bad_row_count": int(self.bad_row_count),
            "min": self.min_value,
            "max": self.max_value,
            "mean": None if mean is None else float(mean),
            "std": std,
            "sample_percentiles": percentiles(sample),
            "sample_iqr_outlier_fraction": sample_iqr_outlier_fraction(sample),
            "row_norm_percentiles": percentiles(row_norms),
        }

        if self.image_row_count:
            summary["image_like"] = {
                "rows_checked": int(self.image_row_count),
                "low_variance_row_fraction": float(
                    self.low_variance_rows / self.image_row_count
                ),
                "blackish_row_fraction": float(
                    self.blackish_rows / self.image_row_count
                ),
            }

        if self.name == "action" and self.finite_count:
            summary["action"] = {
                "saturated_element_fraction": float(
                    self.action_saturated_elements / self.finite_count
                ),
                "out_of_range_element_fraction": float(
                    self.action_out_of_range_elements / self.finite_count
                ),
            }

        return summary


def iter_dataset_chunks(dataset: h5py.Dataset, chunk_rows: int):
    rows = int(dataset.shape[0]) if dataset.shape else 1
    if dataset.shape == ():
        yield np.asarray(dataset[()]).reshape(1)
        return
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        yield dataset[start:end]


def analyze_dataset_object(
    analyzer: ColumnAnalyzer,
    dataset: h5py.Dataset,
    *,
    chunk_rows: int,
) -> None:
    for chunk in iter_dataset_chunks(dataset, chunk_rows):
        analyzer.update(np.asarray(chunk))


def detect_layout(h5_file: h5py.File) -> str:
    if "ep_len" in h5_file and "ep_offset" in h5_file:
        return "flat"
    return "raw_hierarchical"


def dataset_shape_after_first_axis(dataset: h5py.Dataset) -> tuple[int, ...]:
    if dataset.shape == ():
        return ()
    return tuple(int(dim) for dim in dataset.shape[1:])


def make_analyzer(
    key: str,
    dataset: h5py.Dataset,
    args,
    rng: np.random.Generator,
) -> ColumnAnalyzer:
    return ColumnAnalyzer(
        name=key,
        dtype=str(dataset.dtype),
        shape=dataset_shape_after_first_axis(dataset),
        max_sample_values=int(args.sample_values),
        max_sample_row_norms=int(args.sample_row_norms),
        rng=rng,
    )


def analyze_flat(h5_file: h5py.File, args, rng: np.random.Generator):
    ep_len = np.asarray(h5_file["ep_len"][:], dtype=np.int64)
    ep_offset = np.asarray(h5_file["ep_offset"][:], dtype=np.int64)
    total_rows = int(ep_len.sum())

    columns = {}
    for key, item in h5_file.items():
        if key in SKIP_KEYS or key == STATS_GROUP_NAME or not isinstance(item, h5py.Dataset):
            continue
        if item.shape[:1] != (total_rows,):
            continue
        if not is_numeric_dtype(item.dtype):
            continue
        print(f"Analyzing column {key} {item.shape} {item.dtype}", flush=True)
        analyzer = make_analyzer(key, item, args, rng)
        analyze_dataset_object(analyzer, item, chunk_rows=args.chunk_rows)
        columns[key] = analyzer.summarize()

    episodes = flat_episode_summary(h5_file, ep_len, ep_offset)
    jumps = flat_jump_summary(h5_file, ep_len, ep_offset, args, rng)
    return {
        "layout": "flat",
        "episodes": episodes,
        "columns": columns,
        "trajectory_jumps": jumps,
    }


def flat_episode_summary(h5_file: h5py.File, ep_len: np.ndarray, ep_offset: np.ndarray):
    summary = {
        "num_episodes": int(len(ep_len)),
        "total_steps": int(ep_len.sum()),
        "episode_length": {
            "min": int(np.min(ep_len)) if len(ep_len) else None,
            "max": int(np.max(ep_len)) if len(ep_len) else None,
            "mean": float(np.mean(ep_len)) if len(ep_len) else None,
            "std": float(np.std(ep_len)) if len(ep_len) else None,
            "percentiles": percentiles(ep_len.astype(np.float64)),
        },
    }

    if "success" in h5_file:
        success = h5_file["success"]
        episode_success = []
        first_success_steps = []
        for length, offset in zip(ep_len, ep_offset):
            values = np.asarray(success[int(offset) : int(offset + length)])
            values = values.reshape(values.shape[0], -1).any(axis=1)
            positions = np.flatnonzero(values)
            episode_success.append(bool(positions.size))
            if positions.size:
                first_success_steps.append(int(positions[0]))
        summary["success"] = {
            "episode_success_rate": float(np.mean(episode_success))
            if episode_success
            else None,
            "successful_episodes": int(np.sum(episode_success)),
            "first_success_step_percentiles": percentiles(
                np.asarray(first_success_steps, dtype=np.float64)
            ),
        }

    if "step_idx" in h5_file:
        bad = 0
        for length, offset in zip(ep_len, ep_offset):
            expected = np.arange(int(length), dtype=np.int64)
            observed = np.asarray(h5_file["step_idx"][int(offset) : int(offset + length)])
            if observed.shape[0] != expected.shape[0] or not np.array_equal(
                observed.reshape(-1),
                expected,
            ):
                bad += 1
        summary["step_idx_bad_episodes"] = int(bad)

    if "env_idx" in h5_file:
        env_values = np.asarray(h5_file["env_idx"][:]).reshape(-1)
        unique, counts = np.unique(env_values, return_counts=True)
        summary["env_idx_counts"] = {
            str(int(key)): int(value) for key, value in zip(unique, counts)
        }

    return summary


def flat_jump_summary(h5_file: h5py.File, ep_len, ep_offset, args, rng):
    if len(ep_len) == 0:
        return {}

    episode_indices = np.arange(len(ep_len), dtype=np.int64)
    if len(episode_indices) > args.max_jump_episodes:
        episode_indices = np.sort(
            rng.choice(
                episode_indices,
                size=int(args.max_jump_episodes),
                replace=False,
            )
        )

    result = {}
    for key in VECTOR_JUMP_KEYS:
        if key not in h5_file or not isinstance(h5_file[key], h5py.Dataset):
            continue
        dataset = h5_file[key]
        if not is_numeric_dtype(dataset.dtype):
            continue
        row_dim = int(np.prod(dataset.shape[1:])) if len(dataset.shape) > 1 else 1
        if row_dim > args.max_jump_dim:
            continue

        mean_jumps = []
        max_jumps = []
        for ep_idx in episode_indices:
            length = int(ep_len[ep_idx])
            if length < 2:
                continue
            offset = int(ep_offset[ep_idx])
            values = np.asarray(dataset[offset : offset + length])
            values = values.reshape(length, -1).astype(np.float64)
            finite = np.isfinite(values).all(axis=1)
            if finite.sum() < 2:
                continue
            values = values[finite]
            jumps = np.linalg.norm(np.diff(values, axis=0), axis=1)
            if jumps.size:
                mean_jumps.append(float(np.mean(jumps)))
                max_jumps.append(float(np.max(jumps)))

        result[key] = {
            "episodes_checked": int(len(episode_indices)),
            "mean_step_jump_percentiles": percentiles(np.asarray(mean_jumps)),
            "max_step_jump_percentiles": percentiles(np.asarray(max_jumps)),
        }
    return result


def iter_raw_episode_groups(h5_file: h5py.File):
    for env_name, env_group in h5_file.items():
        if not isinstance(env_group, h5py.Group):
            continue
        for episode_name, episode_group in env_group.items():
            if isinstance(episode_group, h5py.Group):
                yield env_name, episode_name, episode_group


def analyze_raw_hierarchical(h5_file: h5py.File, args, rng: np.random.Generator):
    analyzers = {}
    episode_lengths = []
    episode_success = []
    first_success_steps = []

    for env_name, episode_name, episode_group in iter_raw_episode_groups(h5_file):
        if "action" in episode_group:
            length = int(episode_group["action"].shape[0])
            episode_lengths.append(length)
        else:
            length = None

        if "success" in episode_group:
            values = np.asarray(episode_group["success"][:]).reshape(-1)
            positions = np.flatnonzero(values.astype(bool))
            episode_success.append(bool(positions.size))
            if positions.size:
                first_success_steps.append(int(positions[0]))

        for key, dataset in episode_group.items():
            if not isinstance(dataset, h5py.Dataset) or not is_numeric_dtype(dataset.dtype):
                continue
            if key not in analyzers:
                analyzers[key] = make_analyzer(key, dataset, args, rng)
            print(
                f"Analyzing raw {env_name}/{episode_name}/{key} {dataset.shape} {dataset.dtype}",
                flush=True,
            )
            analyze_dataset_object(analyzers[key], dataset, chunk_rows=args.chunk_rows)

    ep_len = np.asarray(episode_lengths, dtype=np.int64)
    episodes = {
        "num_episodes": int(len(ep_len)),
        "total_steps": int(ep_len.sum()) if len(ep_len) else None,
        "episode_length": {
            "min": int(np.min(ep_len)) if len(ep_len) else None,
            "max": int(np.max(ep_len)) if len(ep_len) else None,
            "mean": float(np.mean(ep_len)) if len(ep_len) else None,
            "std": float(np.std(ep_len)) if len(ep_len) else None,
            "percentiles": percentiles(ep_len.astype(np.float64)),
        },
    }
    if episode_success:
        episodes["success"] = {
            "episode_success_rate": float(np.mean(episode_success)),
            "successful_episodes": int(np.sum(episode_success)),
            "first_success_step_percentiles": percentiles(
                np.asarray(first_success_steps, dtype=np.float64)
            ),
        }

    return {
        "layout": "raw_hierarchical",
        "episodes": episodes,
        "columns": {key: analyzer.summarize() for key, analyzer in analyzers.items()},
        "trajectory_jumps": {},
    }


def collect_warnings(summary: dict[str, Any]) -> list[str]:
    warnings = []
    episodes = summary.get("episodes", {})
    ep_len = episodes.get("episode_length", {})
    if ep_len.get("min") is not None and ep_len["min"] <= 1:
        warnings.append(f"Some episodes are extremely short: min length = {ep_len['min']}.")
    success = episodes.get("success")
    if success is not None:
        rate = success.get("episode_success_rate")
        if rate is not None and rate < 0.5:
            warnings.append(f"Episode success rate is low: {rate:.3f}.")
    if episodes.get("step_idx_bad_episodes", 0):
        warnings.append(
            f"{episodes['step_idx_bad_episodes']} episodes have inconsistent step_idx."
        )

    for key, col in summary.get("columns", {}).items():
        finite_ratio = col.get("finite_ratio")
        optional_absent_second_object = (
            key == "object_2_xyz" and finite_ratio is not None and finite_ratio == 0.0
        )
        if finite_ratio is not None and finite_ratio < 1.0:
            if optional_absent_second_object:
                pass
            else:
                warnings.append(
                    f"{key}: finite_ratio={finite_ratio:.8f}, "
                    f"NaNs={col['nan_count']}, +Inf={col['posinf_count']}, -Inf={col['neginf_count']}."
                )
        if col.get("bad_row_count", 0) and not optional_absent_second_object:
            warnings.append(f"{key}: {col['bad_row_count']} rows contain non-finite values.")

        image = col.get("image_like")
        if image is not None and image.get("low_variance_row_fraction", 0.0) > 0.01:
            warnings.append(
                f"{key}: {image['low_variance_row_fraction']:.3%} low-variance frames."
            )
        if image is not None and image.get("blackish_row_fraction", 0.0) > 0.01:
            warnings.append(
                f"{key}: {image['blackish_row_fraction']:.3%} nearly-black frames."
            )

        action = col.get("action")
        if action is not None:
            if action.get("out_of_range_element_fraction", 0.0) > 0:
                warnings.append(
                    "action: values outside [-1, 1] detected "
                    f"({action['out_of_range_element_fraction']:.3%} of elements)."
                )
            if action.get("saturated_element_fraction", 0.0) > 0.20:
                warnings.append(
                    "action: high saturation fraction "
                    f"({action['saturated_element_fraction']:.3%} of elements near +/-1)."
                )

        if key == "depth":
            if col.get("min") is not None and col["min"] < 0:
                warnings.append(f"depth: negative values detected, min={col['min']}.")

    return warnings


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def write_columns_csv(path: Path, columns: dict[str, dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "dtype",
        "shape",
        "rows",
        "finite_ratio",
        "nan_count",
        "posinf_count",
        "neginf_count",
        "bad_row_count",
        "min",
        "max",
        "mean",
        "std",
        "sample_iqr_outlier_fraction",
        "row_norm_p99",
        "row_norm_p99.9",
        "low_variance_row_fraction",
        "blackish_row_fraction",
        "action_saturated_element_fraction",
        "action_out_of_range_element_fraction",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, col in sorted(columns.items()):
            row_norm = col.get("row_norm_percentiles") or {}
            image = col.get("image_like") or {}
            action = col.get("action") or {}
            writer.writerow(
                {
                    "name": key,
                    "dtype": col.get("dtype"),
                    "shape": json.dumps(col.get("shape")),
                    "rows": col.get("rows"),
                    "finite_ratio": col.get("finite_ratio"),
                    "nan_count": col.get("nan_count"),
                    "posinf_count": col.get("posinf_count"),
                    "neginf_count": col.get("neginf_count"),
                    "bad_row_count": col.get("bad_row_count"),
                    "min": col.get("min"),
                    "max": col.get("max"),
                    "mean": col.get("mean"),
                    "std": col.get("std"),
                    "sample_iqr_outlier_fraction": col.get(
                        "sample_iqr_outlier_fraction"
                    ),
                    "row_norm_p99": row_norm.get("p99"),
                    "row_norm_p99.9": row_norm.get("p99.9"),
                    "low_variance_row_fraction": image.get(
                        "low_variance_row_fraction"
                    ),
                    "blackish_row_fraction": image.get("blackish_row_fraction"),
                    "action_saturated_element_fraction": action.get(
                        "saturated_element_fraction"
                    ),
                    "action_out_of_range_element_fraction": action.get(
                        "out_of_range_element_fraction"
                    ),
                }
            )


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def write_markdown(path: Path, dataset_path: Path, summary: dict[str, Any]) -> None:
    warnings = summary.get("warnings", [])
    lines = [
        "# Dataset Sanity Report",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Layout: `{summary.get('layout')}`",
        "",
        "## Warnings",
        "",
    ]
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No major red flags found by these checks.")

    episodes = summary.get("episodes", {})
    lines.extend(
        [
            "",
            "## Episodes",
            "",
            f"- Episodes: {fmt(episodes.get('num_episodes'))}",
            f"- Total steps: {fmt(episodes.get('total_steps'))}",
        ]
    )
    ep_len = episodes.get("episode_length", {})
    lines.append(
        "- Episode length: "
        f"mean={fmt(ep_len.get('mean'))}, std={fmt(ep_len.get('std'))}, "
        f"min={fmt(ep_len.get('min'))}, max={fmt(ep_len.get('max'))}"
    )
    if "success" in episodes:
        success = episodes["success"]
        lines.append(
            "- Success: "
            f"episode_success_rate={fmt(success.get('episode_success_rate'))}, "
            f"successful_episodes={fmt(success.get('successful_episodes'))}"
        )

    lines.extend(
        [
            "",
            "## Columns",
            "",
            "| Column | Shape | Dtype | Finite | Bad rows | Min | Max | Mean | Std | Notes |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for key, col in sorted(summary.get("columns", {}).items()):
        notes = []
        if key == "object_2_xyz" and col.get("finite_ratio") == 0.0:
            notes.append("optional second object absent")
        image = col.get("image_like")
        if image is not None:
            notes.append(
                f"low-var={fmt(image.get('low_variance_row_fraction'))}, "
                f"blackish={fmt(image.get('blackish_row_fraction'))}"
            )
        action = col.get("action")
        if action is not None:
            notes.append(
                f"sat={fmt(action.get('saturated_element_fraction'))}, "
                f"oor={fmt(action.get('out_of_range_element_fraction'))}"
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{key}`",
                    str(col.get("shape")),
                    str(col.get("dtype")),
                    fmt(col.get("finite_ratio")),
                    fmt(col.get("bad_row_count")),
                    fmt(col.get("min")),
                    fmt(col.get("max")),
                    fmt(col.get("mean")),
                    fmt(col.get("std")),
                    "; ".join(notes),
                ]
            )
            + " |"
        )

    jumps = summary.get("trajectory_jumps", {})
    if jumps:
        lines.extend(
            [
                "",
                "## Per-Step Vector Jumps",
                "",
                "| Column | Episodes checked | Mean jump p99 | Max jump p99 | Max jump p99.9 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for key, item in sorted(jumps.items()):
            mean_jump = item.get("mean_step_jump_percentiles") or {}
            max_jump = item.get("max_step_jump_percentiles") or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{key}`",
                        fmt(item.get("episodes_checked")),
                        fmt(mean_jump.get("p99")),
                        fmt(max_jump.get("p99")),
                        fmt(max_jump.get("p99.9")),
                    ]
                )
                + " |"
            )

    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Non-finite values are always a hard red flag.",
            "- High action saturation can make planning look jerky because the data itself contains many clipped expert actions.",
            "- Low-variance or blackish image frames indicate rendering/cache problems.",
            "- Large jump percentiles in `ee_position`, `object_1_xyz`, or `proprio` can reveal simulator/reset discontinuities inside episodes.",
            "- A low success rate in expert-only eval data means first-success goal sampling will discard many episodes.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main():
    args = parse_args()
    dataset_path = resolve_dataset_path(args.dataset, args.cache_dir)
    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else dataset_path.parent / f"{dataset_path.stem}_sanity"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Opening {dataset_path}", flush=True)
    with h5py.File(dataset_path, "r") as h5_file:
        layout = detect_layout(h5_file)
        if layout == "flat":
            summary = analyze_flat(h5_file, args, rng)
        else:
            summary = analyze_raw_hierarchical(h5_file, args, rng)

    summary["dataset_path"] = str(dataset_path)
    summary["warnings"] = collect_warnings(summary)

    write_json(output_dir / "dataset_sanity_summary.json", summary)
    write_columns_csv(output_dir / "column_stats.csv", summary.get("columns", {}))
    write_markdown(output_dir / "dataset_sanity_report.md", dataset_path, summary)

    print(f"\nWrote {output_dir / 'dataset_sanity_report.md'}")
    print(f"Wrote {output_dir / 'dataset_sanity_summary.json'}")
    print(f"Wrote {output_dir / 'column_stats.csv'}")
    if summary["warnings"]:
        print("\nWarnings:")
        for warning in summary["warnings"]:
            print(f"- {warning}")
    else:
        print("\nNo major red flags found by these checks.")


if __name__ == "__main__":
    main()
