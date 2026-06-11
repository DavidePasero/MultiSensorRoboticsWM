"""PCA and t-SNE visualizations for comparing model latent spaces.

The script samples the same dataset clips for every checkpoint, encodes them
with each frozen model, and saves side-by-side PCA/t-SNE plots colored by
available dataset metadata such as trajectory step, contact, success, and
physical distances. Coordinates from independently trained models should be
interpreted within each subplot, not as a shared global coordinate system.
"""

from __future__ import annotations

import argparse
import csv
import h5py
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.experiment_utils import (  # noqa: E402
    build_dataset,
    get_dataset_columns,
    load_cost_model,
    load_experiment_config,
)
from multimodal import get_enabled_modality_configs  # noqa: E402

EPS = 1e-8
DEFAULT_METADATA_KEYS = (
    "episode_idx",
    "step_idx",
    "env_idx",
    "success",
    "bool_contact",
    "ee_position",
    "ee_xyz",
    "object_1_xyz",
    "object_2_xyz",
    "target_pos",
)
DEFAULT_COLOR_KEYS = (
    "episode_idx",
)
EPISODE_COLORS = ("#008B21", "#5F0FF8", "#F5A12E", "#0053E9", "#A50025")
DEFAULT_EXCLUDED_MODEL_PATTERNS = (
    "masked",
    "selfmask",
    "missing_token",
    "latent_reconstruction",
    "latent-reconstruction",
)


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    checkpoint: str


def parse_checkpoint_spec(value: str) -> CheckpointSpec:
    if "=" in value:
        label, checkpoint = value.split("=", 1)
        label = label.strip()
        checkpoint = checkpoint.strip()
    else:
        checkpoint = value.strip()
        label = Path(checkpoint).stem
        label = re.sub(r"_object$", "", label)
        label = re.sub(r"_epoch_\d+$", "", label)
    if not label or not checkpoint:
        raise argparse.ArgumentTypeError(
            "--checkpoint must be CHECKPOINT or LABEL=CHECKPOINT"
        )
    return CheckpointSpec(label=label, checkpoint=checkpoint)


def parse_config_override(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--config must be LABEL=/path/to/config.yaml")
    label, path = value.split("=", 1)
    return label.strip(), Path(path.strip()).expanduser()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PCA and t-SNE latent visualizations for checkpoints."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint_spec,
        required=True,
        help="Checkpoint path/run name. Repeatable. Use LABEL=CHECKPOINT for plot labels.",
    )
    parser.add_argument(
        "--config",
        action="append",
        type=parse_config_override,
        default=[],
        help="Optional per-label config override, e.g. pixels=/path/config.yaml.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name in STABLEWM_HOME.")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument("--num_clips", type=int, default=256)
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help=(
            "Number of episodes to visualize. The default selects exactly five "
            "episodes and colors them with a fixed palette. Set to 0 to use the "
            "old random-clip sampling behavior."
        ),
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Clip length to encode. Defaults to the checkpoint config value.",
    )
    parser.add_argument("--max_points", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-max-iter", type=int, default=1000)
    parser.add_argument(
        "--color-key",
        action="append",
        default=[],
        help="Metadata key to color by. Repeatable. Defaults to useful available keys.",
    )
    parser.add_argument(
        "--keep-dataset-cache",
        action="store_true",
        help="Keep keys_to_cache from checkpoint configs. By default it is cleared.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable per-model latent standardization before PCA/t-SNE.",
    )
    parser.add_argument(
        "--skip-tsne",
        action="store_true",
        help="Only compute PCA. Useful for very large quick checks.",
    )
    parser.add_argument("--point-size", type=float, default=6.0)
    parser.add_argument("--alpha", type=float, default=0.72)
    parser.add_argument(
        "--episode-min-alpha",
        type=float,
        default=0.08,
        help=(
            "Minimum opacity for the first step of each episode in episode_idx "
            "plots. The opacity then increases linearly with step_idx up to --alpha."
        ),
    )
    parser.add_argument(
        "--episode-colors",
        nargs="+",
        default=list(EPISODE_COLORS),
        help="Colors used for episode_idx plots when --num-episodes is enabled.",
    )
    parser.add_argument(
        "--include-masked-models",
        action="store_true",
        help="Do not filter out masked/selfmask/latent-reconstruction variants.",
    )
    parser.add_argument(
        "--planning-log",
        action="append",
        type=Path,
        default=[],
        help=(
            "Planning run log to use for prioritizing visualized episodes. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--planning-log-dir",
        action="append",
        type=Path,
        default=[],
        help="Directory searched recursively for planning *.log files.",
    )
    parser.add_argument(
        "--planning-model-filter",
        action="append",
        default=[],
        help=(
            "Only use planning logs whose model label or policy contains this "
            "substring. Repeatable. Defaults to all logs."
        ),
    )
    parser.add_argument(
        "--planning-outcome",
        choices=("mixed", "success", "failure", "any"),
        default="mixed",
        help=(
            "How planning logs choose episodes. mixed selects successful episodes "
            "first and keeps some failed ones for contrast."
        ),
    )
    parser.add_argument(
        "--planning-default-goal-offset-steps",
        type=int,
        default=20,
        help=(
            "Fallback goal offset used only for planning logs that contain "
            "METRICS_JSON but not the original Hydra command."
        ),
    )
    return parser.parse_args()


def is_excluded_model_variant(spec: CheckpointSpec) -> bool:
    text = f"{spec.label} {spec.checkpoint}".lower()
    return any(pattern in text for pattern in DEFAULT_EXCLUDED_MODEL_PATTERNS)


def enabled_model_sources(cfg) -> set[str]:
    if not OmegaConf.select(cfg, "obs_encoder.modalities"):
        return {"pixels", "action"}
    sources = {"action"}
    for name, mod_cfg in get_enabled_modality_configs(cfg.obs_encoder).items():
        sources.add(str(mod_cfg.get("source", name)))
    return sources


def configure_dataset(cfg, dataset_name: str, num_steps: int | None, keep_cache: bool):
    with open_dict(cfg):
        cfg.data.dataset.name = dataset_name
        if num_steps is not None:
            cfg.data.dataset.num_steps = int(num_steps)
        if not keep_cache:
            cfg.data.dataset.keys_to_cache = []


def metadata_keys_to_load(cfg, available_columns: list[str]) -> tuple[list[str], list[str]]:
    model_sources = enabled_model_sources(cfg)
    keys = [
        key
        for key in DEFAULT_METADATA_KEYS
        if key in available_columns and key not in model_sources
    ]
    existing = set(cfg.data.dataset.keys_to_load)
    extra = [key for key in keys if key not in existing]
    return extra, keys


def sample_indices(dataset_len: int, count: int, seed: int) -> list[int]:
    if dataset_len <= 0:
        raise ValueError("Dataset produced no valid clips.")
    count = min(int(count), dataset_len)
    rng = np.random.default_rng(seed)
    return rng.choice(dataset_len, size=count, replace=False).tolist()


def sample_episode_clip_indices(
    dataset,
    *,
    episode_count: int,
    seed: int,
    max_dataset_len: int | None = None,
    preferred_episodes: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    column_names = set(getattr(dataset, "column_names", []))
    episode_key = "episode_idx" if "episode_idx" in column_names else "ep_idx"
    if episode_key not in column_names:
        raise ValueError(
            "Episode-based latent visualization requires an episode_idx or ep_idx "
            "column in the dataset. Use --num-episodes 0 to fall back to random clips."
        )

    clip_count = len(dataset)
    if max_dataset_len is not None:
        clip_count = min(clip_count, int(max_dataset_len))
    if clip_count <= 0:
        raise ValueError("Dataset produced no valid clips.")

    row_episode_idx = np.asarray(dataset.get_col_data(episode_key)).reshape(-1)
    if all(
        hasattr(dataset, attr)
        for attr in (
            "clip_shard_indices",
            "clip_episode_indices",
            "clip_start_indices",
            "_shards",
            "_row_ranges",
        )
    ):
        clip_global_rows = np.empty(clip_count, dtype=np.int64)
        clip_shard_indices = np.asarray(dataset.clip_shard_indices[:clip_count])
        clip_episode_indices = np.asarray(dataset.clip_episode_indices[:clip_count])
        clip_start_indices = np.asarray(dataset.clip_start_indices[:clip_count])

        for shard_idx, shard in enumerate(dataset._shards):
            mask = clip_shard_indices == shard_idx
            if not np.any(mask):
                continue
            local_eps = clip_episode_indices[mask].astype(np.int64)
            starts = clip_start_indices[mask].astype(np.int64)
            row_base = int(dataset._row_ranges[shard_idx][0])
            clip_global_rows[mask] = row_base + shard["offsets"][local_eps] + starts
    elif all(hasattr(dataset, attr) for attr in ("clip_indices", "offsets")):
        clip_indices = list(dataset.clip_indices[:clip_count])
        clip_global_rows = np.asarray(
            [int(dataset.offsets[int(ep_idx)] + int(start)) for ep_idx, start in clip_indices],
            dtype=np.int64,
        )
    else:
        raise ValueError(
            "Episode-based latent visualization requires clip index metadata. "
            "Use --num-episodes 0 to fall back to random clips."
        )

    clip_episode_ids = row_episode_idx[clip_global_rows]
    unique_episodes = np.unique(clip_episode_ids)
    if unique_episodes.size == 0:
        raise ValueError("No episodes were available for visualization.")

    rng = np.random.default_rng(seed)
    count = min(int(episode_count), int(unique_episodes.size))
    if preferred_episodes:
        available = set(int(ep) for ep in unique_episodes.tolist())
        selected = []
        for ep in preferred_episodes:
            ep = int(ep)
            if ep in available and ep not in selected:
                selected.append(ep)
            if len(selected) >= count:
                break

        if len(selected) < count:
            remaining = np.asarray(
                [int(ep) for ep in unique_episodes.tolist() if int(ep) not in selected],
                dtype=np.int64,
            )
            if remaining.size > 0:
                fill = rng.choice(
                    remaining,
                    size=min(count - len(selected), int(remaining.size)),
                    replace=False,
                )
                selected.extend(int(ep) for ep in fill.tolist())
        selected_episodes = np.asarray(selected[:count], dtype=np.int64)
    else:
        selected_episodes = np.sort(
            rng.choice(unique_episodes, size=count, replace=False)
        )
    indices = np.flatnonzero(np.isin(clip_episode_ids, selected_episodes)).tolist()
    if not indices:
        raise ValueError("Selected episodes produced no valid clips.")

    return indices, [int(ep) for ep in selected_episodes.tolist()]


def collect_planning_log_paths(args: argparse.Namespace) -> list[Path]:
    paths = [path.expanduser() for path in args.planning_log]
    for directory in args.planning_log_dir:
        directory = directory.expanduser()
        if directory.exists():
            paths.extend(
                path
                for path in sorted(directory.rglob("*.log"))
                if path.name != "planning_runs.log"
            )

    unique = []
    seen = set()
    for path in paths:
        path = path.resolve()
        if path.name == "planning_runs.log":
            continue
        if path in seen or not path.exists():
            continue
        seen.add(path)
        unique.append(path)
    return unique


def parse_planning_log(path: Path, *, default_goal_offset_steps: int) -> dict | None:
    text = path.read_text(errors="replace")
    command_match = re.search(r"^Command:\s*(.+)$", text, flags=re.MULTILINE)
    metrics_match = re.findall(r"^METRICS_JSON=(.+)$", text, flags=re.MULTILINE)
    if not metrics_match:
        return None

    overrides = {}
    if command_match is not None:
        for token in shlex.split(command_match.group(1)):
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            overrides[key] = value

    header_match = re.search(
        r"^=== TASK=(?P<task>.*?) MODEL=(?P<model>.*?) RUN=(?P<run>\d+) SEED=(?P<seed>\d+) ===$",
        text,
        flags=re.MULTILINE,
    )
    filename_match = re.match(
        r"(?P<task>.*?)__(?P<model>.*?)__run_(?P<run>\d+)\.log$",
        path.name,
    )
    run_id = (
        int(header_match.group("run"))
        if header_match
        else int(filename_match.group("run"))
        if filename_match
        else 1
    )
    metrics = json.loads(metrics_match[-1])
    return {
        "path": str(path),
        "task": overrides.get(
            "world.metaworld_env_name",
            header_match.group("task")
            if header_match
            else filename_match.group("task")
            if filename_match
            else None,
        ),
        "model": (
            header_match.group("model")
            if header_match
            else filename_match.group("model")
            if filename_match
            else None
        ),
        "policy": overrides.get("policy"),
        "dataset": overrides.get("eval.dataset_name"),
        "seed": int(
            overrides.get(
                "seed",
                header_match.group("seed") if header_match else 41 + run_id,
            )
        ),
        "num_eval": int(overrides.get("eval.num_eval", len(metrics.get("episode_successes", [])))),
        "goal_sampling": overrides.get("eval.goal_sampling", "first_success"),
        "goal_success_key": overrides.get("eval.goal_success_key", "success"),
        "goal_offset_steps": int(
            overrides.get("eval.goal_offset_steps", default_goal_offset_steps)
        ),
        "metrics": metrics,
    }


def resolve_planning_env_idx(dataset, task_name: str | None) -> int | None:
    if task_name is None or "env_idx" not in getattr(dataset, "column_names", []):
        return None
    h5_path = getattr(dataset, "h5_path", None)
    if h5_path is None:
        return None
    with h5py.File(h5_path, "r") as h5_file:
        names_json = h5_file.attrs.get("env_names_json", None)
    if names_json is None:
        return None
    env_names = json.loads(names_json)
    return int(env_names.index(task_name)) if task_name in env_names else None


def recover_first_success_eval_pairs(dataset, record: dict) -> tuple[np.ndarray, np.ndarray]:
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = np.asarray(dataset.get_col_data(col_name)).reshape(-1)
    step_idx = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    success_values = np.asarray(dataset.get_col_data(record["goal_success_key"]))
    successes = success_values.reshape(success_values.shape[0], -1).any(axis=1)
    env_idx = None
    resolved_env_idx = resolve_planning_env_idx(dataset, record.get("task"))
    if resolved_env_idx is not None and "env_idx" in dataset.column_names:
        env_idx = np.asarray(dataset.get_col_data("env_idx")).reshape(-1)

    candidate_episodes = []
    candidate_starts = []
    start_offset = int(record["goal_offset_steps"])
    for ep_id in np.unique(episode_idx):
        mask = episode_idx == ep_id
        if env_idx is not None:
            mask &= env_idx == resolved_env_idx
        if not np.any(mask):
            continue

        steps = step_idx[mask].astype(np.int64)
        ep_successes = successes[mask]
        order = np.argsort(steps)
        steps = steps[order]
        ep_successes = ep_successes[order]
        success_positions = np.flatnonzero(ep_successes)
        if success_positions.size == 0:
            continue

        goal_step = int(steps[success_positions[0]])
        start_step = goal_step - start_offset
        if start_step < int(steps[0]) or not np.any(steps == start_step):
            continue
        candidate_episodes.append(int(ep_id))
        candidate_starts.append(start_step)

    if len(candidate_episodes) < int(record["num_eval"]):
        raise ValueError(
            f"Could not recover enough planning eval episodes from {record['path']}: "
            f"found {len(candidate_episodes)}, need {record['num_eval']}."
        )

    rng = np.random.default_rng(int(record["seed"]))
    selected = np.sort(
        rng.choice(
            len(candidate_episodes),
            size=int(record["num_eval"]),
            replace=False,
        )
    )
    return (
        np.asarray(candidate_episodes, dtype=np.int64)[selected],
        np.asarray(candidate_starts, dtype=np.int64)[selected],
    )


def recover_fixed_offset_eval_pairs(dataset, record: dict) -> tuple[np.ndarray, np.ndarray]:
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = np.asarray(dataset.get_col_data(col_name)).reshape(-1)
    step_idx = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    env_idx = None
    resolved_env_idx = resolve_planning_env_idx(dataset, record.get("task"))
    if resolved_env_idx is not None and "env_idx" in dataset.column_names:
        env_idx = np.asarray(dataset.get_col_data("env_idx")).reshape(-1)

    goal_offset = int(record["goal_offset_steps"]) + 1
    max_start_by_episode = {}
    for ep_id in np.unique(episode_idx):
        mask = episode_idx == ep_id
        if env_idx is not None:
            mask &= env_idx == resolved_env_idx
        if not np.any(mask):
            continue
        max_start_by_episode[int(ep_id)] = int(np.max(step_idx[mask]) + 1 - goal_offset - 1)

    valid_mask = np.asarray(
        [
            int(step) <= max_start_by_episode.get(int(ep), -1)
            for ep, step in zip(episode_idx, step_idx)
        ],
        dtype=bool,
    )
    if env_idx is not None:
        valid_mask &= env_idx == resolved_env_idx
    valid_indices = np.nonzero(valid_mask)[0]
    if len(valid_indices) < int(record["num_eval"]):
        raise ValueError(
            f"Could not recover enough fixed-offset planning rows from {record['path']}: "
            f"found {len(valid_indices)}, need {record['num_eval']}."
        )

    rng = np.random.default_rng(int(record["seed"]))
    selected_rows = np.sort(
        valid_indices[
            rng.choice(len(valid_indices), size=int(record["num_eval"]), replace=False)
        ]
    )
    rows = dataset.get_row_data(selected_rows.tolist())
    return np.asarray(rows[col_name], dtype=np.int64), np.asarray(rows["step_idx"], dtype=np.int64)


def planning_episode_priority(
    *,
    dataset,
    dataset_name: str,
    log_paths: list[Path],
    model_filters: list[str],
    outcome: str,
    max_episodes: int,
    default_goal_offset_steps: int,
    excluded_model_patterns: tuple[str, ...] = (),
) -> tuple[list[int], dict[int, dict]]:
    if not log_paths:
        return [], {}

    filters = [item.lower() for item in model_filters]
    stats: dict[int, dict] = {}
    for path in log_paths:
        record = parse_planning_log(
            path,
            default_goal_offset_steps=default_goal_offset_steps,
        )
        if record is None:
            continue
        if record.get("dataset") and record["dataset"] != dataset_name:
            continue
        searchable = " ".join(
            str(record.get(key) or "") for key in ("model", "policy", "task", "path")
        ).lower()
        if any(pattern in searchable for pattern in excluded_model_patterns):
            continue
        if filters and not any(item in searchable for item in filters):
            continue

        try:
            if record["goal_sampling"] == "first_success":
                episodes, starts = recover_first_success_eval_pairs(dataset, record)
            else:
                episodes, starts = recover_fixed_offset_eval_pairs(dataset, record)
        except (KeyError, ValueError) as exc:
            print(f"Skipping planning log {path}: {exc}", flush=True)
            continue
        successes = list(record["metrics"].get("episode_successes", []))
        for ep, start, success in zip(episodes, starts, successes):
            ep = int(ep)
            entry = stats.setdefault(
                ep,
                {
                    "episode_idx": ep,
                    "success_count": 0,
                    "failure_count": 0,
                    "total_count": 0,
                    "starts": [],
                    "models": set(),
                },
            )
            entry["success_count"] += int(bool(success))
            entry["failure_count"] += int(not bool(success))
            entry["total_count"] += 1
            entry["starts"].append(int(start))
            if record.get("model"):
                entry["models"].add(str(record["model"]))

    if not stats:
        return [], {}

    def success_key(item):
        ep, entry = item
        return (-entry["success_count"], -entry["total_count"], ep)

    def failure_key(item):
        ep, entry = item
        return (-entry["failure_count"], -entry["total_count"], ep)

    def any_key(item):
        ep, entry = item
        return (-entry["total_count"], -entry["success_count"], ep)

    success_pool = sorted(
        [(ep, entry) for ep, entry in stats.items() if entry["success_count"] > 0],
        key=success_key,
    )
    failure_pool = sorted(
        [(ep, entry) for ep, entry in stats.items() if entry["failure_count"] > 0],
        key=failure_key,
    )

    ordered = []
    if outcome == "success":
        ordered = [ep for ep, _entry in success_pool]
    elif outcome == "failure":
        ordered = [ep for ep, _entry in failure_pool]
    elif outcome == "any":
        ordered = [ep for ep, _entry in sorted(stats.items(), key=any_key)]
    else:
        target_success = max_episodes if not failure_pool else (max_episodes + 1) // 2
        ordered.extend(ep for ep, _entry in success_pool[:target_success])
        ordered.extend(ep for ep, _entry in failure_pool if ep not in ordered)
        ordered.extend(
            ep
            for ep, _entry in sorted(stats.items(), key=any_key)
            if ep not in ordered
        )

    for entry in stats.values():
        entry["models"] = sorted(entry["models"])
    return ordered[:max_episodes], stats


def planning_episode_labels(stats: dict[int, dict]) -> dict[int, str]:
    labels = {}
    for ep, entry in stats.items():
        labels[int(ep)] = (
            f"ep {int(ep)} "
            f"S={int(entry['success_count'])} F={int(entry['failure_count'])}"
        )
    return labels


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(val, device) for key, val in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def sanitize_batch(batch: dict) -> dict:
    copied = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.clone()
            if torch.is_floating_point(value):
                value = torch.nan_to_num(value, 0.0)
        copied[key] = value
    return copied


def flatten_sequence(value: torch.Tensor, expected_steps: int | None = None):
    if not torch.is_tensor(value) or value.ndim < 2:
        return None
    if expected_steps is not None and value.shape[1] != expected_steps:
        return None
    return value.detach().cpu().reshape(value.shape[0] * value.shape[1], *value.shape[2:])


def append_metadata(
    storage: dict[str, list[np.ndarray]],
    batch: dict,
    *,
    num_steps: int,
):
    for key in DEFAULT_METADATA_KEYS:
        flat = flatten_sequence(batch.get(key), expected_steps=num_steps)
        if flat is None:
            continue
        arr = flat.numpy()
        if arr.ndim > 1 and int(np.prod(arr.shape[1:])) == 1:
            arr = arr.reshape(arr.shape[0])
        storage.setdefault(key, []).append(arr)


def finalize_metadata(storage: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    metadata = {
        key: np.concatenate(chunks, axis=0)
        for key, chunks in storage.items()
        if chunks
    }

    ee = metadata.get("ee_position")
    if ee is None:
        ee = metadata.get("ee_xyz")
    obj = metadata.get("object_1_xyz")
    target = metadata.get("target_pos")

    if ee is not None and obj is not None:
        metadata["ee_object_distance"] = rowwise_distance(ee, obj)
    if ee is not None and target is not None:
        metadata["ee_target_distance"] = rowwise_distance(ee, target)

    return metadata


def rowwise_distance(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(len(a), -1)
    b = np.asarray(b, dtype=np.float64).reshape(len(b), -1)
    dims = min(a.shape[1], b.shape[1])
    out = np.full(len(a), np.nan, dtype=np.float64)
    if dims <= 0:
        return out
    diff = a[:, :dims] - b[:, :dims]
    valid = np.isfinite(diff).all(axis=1)
    out[valid] = np.linalg.norm(diff[valid], axis=1)
    return out


@torch.no_grad()
def extract_latents_for_model(
    *,
    spec: CheckpointSpec,
    dataset,
    indices: list[int],
    cache_dir: Path | None,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    collect_metadata: bool,
):
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = load_cost_model(spec.checkpoint, cache_dir=cache_dir).to(device).eval()
    model.requires_grad_(False)

    latents = []
    metadata_storage = {}
    nonfinite_count = 0
    total_values = 0

    for batch in loader:
        batch_device = move_to_device(sanitize_batch(batch), device)
        outputs = model.encode(batch_device)
        emb = outputs["emb"]
        nonfinite_count += int((~torch.isfinite(emb)).sum().item())
        total_values += int(emb.numel())
        emb = torch.nan_to_num(emb, 0.0)
        latents.append(emb.detach().cpu().float().reshape(-1, emb.shape[-1]))

        if collect_metadata:
            append_metadata(
                metadata_storage,
                batch,
                num_steps=int(emb.shape[1]),
            )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    metadata = finalize_metadata(metadata_storage) if collect_metadata else {}
    return torch.cat(latents, dim=0).numpy(), metadata, nonfinite_count, total_values


def standardize_latents(latents: np.ndarray) -> np.ndarray:
    latents = np.asarray(latents, dtype=np.float32)
    latents = np.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0)
    mean = latents.mean(axis=0, keepdims=True)
    std = latents.std(axis=0, keepdims=True)
    return (latents - mean) / np.maximum(std, EPS)


def select_points(num_points: int, max_points: int, seed: int) -> np.ndarray:
    count = min(int(max_points), int(num_points))
    if count <= 0:
        raise ValueError("No latent points available for visualization.")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_points, size=count, replace=False))


def select_points_balanced_by_episode(
    metadata: dict[str, np.ndarray],
    total_points: int,
    max_points: int,
    seed: int,
) -> np.ndarray:
    if "episode_idx" not in metadata:
        return select_points(total_points, max_points, seed)

    values = np.asarray(metadata["episode_idx"][:total_points]).reshape(-1)
    unique = np.unique(values)
    if unique.size == 0:
        return select_points(total_points, max_points, seed)

    rng = np.random.default_rng(seed)
    max_points = min(int(max_points), int(total_points))
    per_episode = max(1, max_points // int(unique.size))
    selected = []
    selected_set = set()

    for ep in unique:
        candidates = np.flatnonzero(values == ep)
        if candidates.size == 0:
            continue
        take = min(per_episode, int(candidates.size))
        chosen = rng.choice(candidates, size=take, replace=False)
        selected.extend(chosen.tolist())
        selected_set.update(int(idx) for idx in chosen.tolist())

    if len(selected) < max_points:
        remaining = np.asarray(
            [idx for idx in range(total_points) if idx not in selected_set],
            dtype=np.int64,
        )
        if remaining.size > 0:
            take = min(max_points - len(selected), int(remaining.size))
            selected.extend(rng.choice(remaining, size=take, replace=False).tolist())

    return np.asarray(sorted(selected[:max_points]), dtype=np.int64)


def compute_pca(latents: np.ndarray, seed: int):
    if len(latents) < 2:
        return None, None
    reducer = PCA(n_components=2, random_state=seed)
    coords = reducer.fit_transform(latents)
    return coords, reducer.explained_variance_ratio_.astype(float).tolist()


def compute_tsne(latents: np.ndarray, perplexity: float, max_iter: int, seed: int):
    if len(latents) < 3:
        return None
    safe_perplexity = min(float(perplexity), max(1.0, (len(latents) - 1) / 3.0))
    reducer = TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=int(max_iter),
        random_state=seed,
    )
    return reducer.fit_transform(latents)


def available_color_keys(metadata: dict[str, np.ndarray], requested: list[str]) -> list[str]:
    if requested:
        missing = [key for key in requested if key not in metadata]
        if missing:
            print(f"Skipping unavailable color keys: {', '.join(missing)}")
        return [key for key in requested if key in metadata]
    return [key for key in DEFAULT_COLOR_KEYS if key in metadata]


def color_values(metadata: dict[str, np.ndarray], key: str, selection: np.ndarray):
    values = np.asarray(metadata[key])[selection]
    if values.ndim > 1:
        values = values.reshape(values.shape[0], -1)
        if values.shape[1] == 1:
            values = values[:, 0]
        else:
            values = np.linalg.norm(values.astype(np.float64), axis=1)
    return values


def is_binary(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values.astype(np.float64, copy=False))]
    if finite.size == 0:
        return False
    unique = np.unique(finite)
    return unique.size <= 2 and set(unique.tolist()).issubset({0, 1, 0.0, 1.0})


def plot_projection_grid(
    projections: dict[str, np.ndarray],
    *,
    method: str,
    color_key: str,
    values: np.ndarray,
    output_path: Path,
    point_size: float,
    alpha: float,
    explained_variance: dict[str, list[float]] | None = None,
    episode_colors: list[str] | None = None,
    episode_alpha_values: np.ndarray | None = None,
    episode_order: list[int] | None = None,
    episode_labels: dict[int, str] | None = None,
):
    labels = list(projections.keys())
    ncols = min(3, len(labels))
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.0 * ncols, 4.4 * nrows),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    values_numeric = values.astype(np.float64, copy=False)
    finite = np.isfinite(values_numeric)
    episode_colors = episode_colors or list(EPISODE_COLORS)
    episode_plot = (
        color_key == "episode_idx"
        and finite.any()
        and np.unique(values_numeric[finite]).size <= len(episode_colors)
    )
    if episode_plot and episode_order is not None:
        available_episode_ids = set(int(ep) for ep in np.unique(values_numeric[finite]))
        episode_ids = [int(ep) for ep in episode_order if int(ep) in available_episode_ids]
    else:
        episode_ids = np.unique(values_numeric[finite]).tolist() if episode_plot else []
    binary = False if episode_plot else is_binary(values)
    cmap = "coolwarm" if binary else "viridis"
    vmin = 0.0 if binary else None
    vmax = 1.0 if binary else None
    if not binary and finite.any():
        vmin = float(np.nanmin(values[finite]))
        vmax = float(np.nanmax(values[finite]))
        if abs(vmax - vmin) < EPS:
            vmax = vmin + 1.0

    scatter = None
    for ax, label in zip(axes_flat, labels):
        coords = projections[label]
        if (~finite).any():
            ax.scatter(
                coords[~finite, 0],
                coords[~finite, 1],
                s=point_size,
                c="#c7c7c7",
                alpha=0.35,
                linewidths=0,
            )
        if episode_plot:
            for ep, color in zip(episode_ids, episode_colors):
                ep_mask = finite & (values_numeric == ep)
                if not np.any(ep_mask):
                    continue
                base_color = np.asarray(mcolors.to_rgba(color), dtype=np.float64)
                point_colors = np.tile(base_color, (int(ep_mask.sum()), 1))
                if episode_alpha_values is not None:
                    point_alphas = np.asarray(episode_alpha_values[ep_mask], dtype=float)
                    point_colors[:, 3] = np.clip(point_alphas, 0.0, 1.0)
                else:
                    point_colors[:, 3] = alpha
                order = np.argsort(point_colors[:, 3])
                ep_coords = coords[ep_mask]
                ax.scatter(
                    ep_coords[order, 0],
                    ep_coords[order, 1],
                    s=point_size,
                    c=point_colors[order],
                    linewidths=0,
                )
                ax.scatter(
                    [],
                    [],
                    s=point_size,
                    c=color,
                    label=(
                        episode_labels.get(int(ep), f"episode {int(ep)}")
                        if episode_labels is not None
                        else f"episode {int(ep)}"
                    ),
                    alpha=1.0,
                    linewidths=0,
                )
        else:
            scatter = ax.scatter(
                coords[finite, 0],
                coords[finite, 1],
                s=point_size,
                c=values[finite],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=alpha,
                linewidths=0,
            )
        title = label
        x_label = f"{method.upper()} 1"
        y_label = f"{method.upper()} 2"
        if method == "pca" and explained_variance is not None:
            variance = explained_variance.get(label)
            if variance is not None and len(variance) >= 2:
                pc1 = 100.0 * float(variance[0])
                pc2 = 100.0 * float(variance[1])
                title = f"{label}\nPC1+PC2: {pc1 + pc2:.1f}%"
                x_label = f"PC1 ({pc1:.1f}%)"
                y_label = f"PC2 ({pc2:.1f}%)"
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, linewidth=0.25, alpha=0.35)
        if episode_plot:
            ax.legend(loc="best", fontsize=7, frameon=False)

    for ax in axes_flat[len(labels) :]:
        ax.axis("off")

    if scatter is not None and not episode_plot:
        cbar = fig.colorbar(scatter, ax=axes_flat[: len(labels)], shrink=0.82)
        cbar.set_label(color_key)
        if binary:
            cbar.set_ticks([0, 1])

    fig.suptitle(f"{method.upper()} latent visualization colored by {color_key}")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def episode_step_alphas(
    metadata: dict[str, np.ndarray],
    selection: np.ndarray,
    *,
    max_alpha: float,
    min_alpha: float,
) -> np.ndarray | None:
    if "episode_idx" not in metadata or "step_idx" not in metadata:
        return None

    episodes = np.asarray(metadata["episode_idx"])[selection].reshape(-1)
    steps = np.asarray(metadata["step_idx"])[selection].reshape(-1).astype(np.float64)
    alphas = np.full(len(selection), float(max_alpha), dtype=np.float64)
    min_alpha = float(np.clip(min_alpha, 0.0, max_alpha))

    for ep in np.unique(episodes):
        mask = episodes == ep
        ep_steps = steps[mask]
        finite = np.isfinite(ep_steps)
        if not finite.any():
            continue
        lo = float(np.nanmin(ep_steps[finite]))
        hi = float(np.nanmax(ep_steps[finite]))
        if abs(hi - lo) < EPS:
            norm = np.ones_like(ep_steps, dtype=np.float64)
        else:
            norm = (ep_steps - lo) / (hi - lo)
        norm = np.nan_to_num(norm, nan=1.0, posinf=1.0, neginf=0.0)
        alphas[mask] = min_alpha + np.clip(norm, 0.0, 1.0) * (max_alpha - min_alpha)

    return alphas


def write_projection_csv(
    path: Path,
    projections_by_method: dict[str, dict[str, np.ndarray]],
    metadata: dict[str, np.ndarray],
    selection: np.ndarray,
    color_keys: list[str],
):
    fieldnames = ["method", "model", "point_index", "x", "y", *color_keys]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, projections in projections_by_method.items():
            for label, coords in projections.items():
                for local_idx, point_idx in enumerate(selection):
                    row = {
                        "method": method,
                        "model": label,
                        "point_index": int(point_idx),
                        "x": float(coords[local_idx, 0]),
                        "y": float(coords[local_idx, 1]),
                    }
                    for key in color_keys:
                        value = np.asarray(metadata[key])[point_idx]
                        if np.asarray(value).size == 1:
                            row[key] = float(np.asarray(value).reshape(-1)[0])
                        else:
                            row[key] = json.dumps(np.asarray(value).tolist())
                    writer.writerow(row)


def main():
    args = parse_args()
    if not args.include_masked_models:
        kept = []
        skipped = []
        for spec in args.checkpoint:
            if is_excluded_model_variant(spec):
                skipped.append(spec.label)
            else:
                kept.append(spec)
        if skipped:
            print(
                "Skipping masked/imputer variants: " + ", ".join(skipped),
                flush=True,
            )
        args.checkpoint = kept
    if not args.checkpoint:
        raise ValueError("No checkpoints left to visualize after model filtering.")
    if len({spec.label for spec in args.checkpoint}) != len(args.checkpoint):
        raise ValueError("Checkpoint labels must be unique.")
    if args.num_episodes > len(args.episode_colors):
        raise ValueError(
            f"--num-episodes={args.num_episodes} needs at least that many "
            f"--episode-colors, got {len(args.episode_colors)}."
        )

    cache_dir = args.cache_dir.expanduser() if args.cache_dir is not None else None
    config_overrides = dict(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    available_columns = get_dataset_columns(args.dataset, cache_dir)
    prepared = []
    min_dataset_len = None

    for spec in args.checkpoint:
        cfg, config_path = load_experiment_config(
            checkpoint=spec.checkpoint,
            cache_dir=cache_dir,
            config_path=config_overrides.get(spec.label),
            dataset_name=args.dataset,
        )
        configure_dataset(cfg, args.dataset, args.num_steps, args.keep_dataset_cache)
        extra_keys, passthrough_keys = metadata_keys_to_load(cfg, available_columns)
        dataset = build_dataset(
            cfg,
            cache_dir,
            extra_keys_to_load=extra_keys,
            passthrough_keys=passthrough_keys,
        )
        min_dataset_len = (
            len(dataset)
            if min_dataset_len is None
            else min(min_dataset_len, len(dataset))
        )
        prepared.append(
            {
                "spec": spec,
                "cfg": cfg,
                "config_path": config_path,
                "dataset": dataset,
                "passthrough_keys": passthrough_keys,
            }
        )

    planning_log_paths = collect_planning_log_paths(args)
    planning_preferred_episodes: list[int] = []
    planning_stats: dict[int, dict] = {}
    if args.num_episodes > 0 and planning_log_paths:
        planning_preferred_episodes, planning_stats = planning_episode_priority(
            dataset=prepared[0]["dataset"],
            dataset_name=args.dataset,
            log_paths=planning_log_paths,
            model_filters=args.planning_model_filter,
            outcome=args.planning_outcome,
            max_episodes=args.num_episodes,
            default_goal_offset_steps=args.planning_default_goal_offset_steps,
            excluded_model_patterns=(
                ()
                if args.include_masked_models
                else DEFAULT_EXCLUDED_MODEL_PATTERNS
            ),
        )
        if planning_preferred_episodes:
            print(
                "Planning-prioritized episodes: "
                + ", ".join(str(ep) for ep in planning_preferred_episodes),
                flush=True,
            )
        else:
            print(
                "No matching planning episodes recovered; falling back to random episodes.",
                flush=True,
            )

    selected_episodes = None
    if args.num_episodes > 0:
        indices, selected_episodes = sample_episode_clip_indices(
            prepared[0]["dataset"],
            episode_count=args.num_episodes,
            seed=args.seed,
            max_dataset_len=int(min_dataset_len),
            preferred_episodes=planning_preferred_episodes,
        )
        print(
            "Selected episodes: " + ", ".join(str(ep) for ep in selected_episodes),
            flush=True,
        )
    else:
        indices = sample_indices(int(min_dataset_len), args.num_clips, args.seed)
    latents_by_model = {}
    metadata = None
    model_summaries = {}

    for i, item in enumerate(prepared):
        spec = item["spec"]
        print(f"Encoding {spec.label} from {spec.checkpoint} ...", flush=True)
        latents, model_metadata, nonfinite_count, total_values = extract_latents_for_model(
            spec=spec,
            dataset=item["dataset"],
            indices=indices,
            cache_dir=cache_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collect_metadata=(i == 0),
        )
        if metadata is None:
            metadata = model_metadata
        latents_by_model[spec.label] = latents
        model_summaries[spec.label] = {
            "checkpoint": spec.checkpoint,
            "config_path": str(item["config_path"]),
            "latent_dim": int(latents.shape[1]),
            "num_points_before_plot_subsample": int(latents.shape[0]),
            "nonfinite_latent_values": int(nonfinite_count),
            "latent_value_count": int(total_values),
            "metadata_keys_loaded": item["passthrough_keys"],
        }

    assert metadata is not None
    total_points = min(latents.shape[0] for latents in latents_by_model.values())
    if args.num_episodes > 0:
        selection = select_points_balanced_by_episode(
            metadata,
            total_points,
            args.max_points,
            args.seed,
        )
    else:
        selection = select_points(total_points, args.max_points, args.seed)
    color_keys = available_color_keys(metadata, args.color_key)
    if not color_keys:
        raise ValueError(
            "No requested/default color keys were available in the dataset metadata."
        )

    projections_by_method: dict[str, dict[str, np.ndarray]] = {"pca": {}}
    pca_variance = {}
    for label, latents in latents_by_model.items():
        selected = latents[:total_points][selection]
        if not args.no_standardize:
            selected = standardize_latents(selected)
        coords, variance = compute_pca(selected, args.seed)
        if coords is None:
            raise ValueError("Need at least two points for PCA.")
        projections_by_method["pca"][label] = coords
        pca_variance[label] = variance

    if not args.skip_tsne:
        projections_by_method["tsne"] = {}
        for label, latents in latents_by_model.items():
            selected = latents[:total_points][selection]
            if not args.no_standardize:
                selected = standardize_latents(selected)
            coords = compute_tsne(
                selected,
                perplexity=args.tsne_perplexity,
                max_iter=args.tsne_max_iter,
                seed=args.seed,
            )
            if coords is not None:
                projections_by_method["tsne"][label] = coords

    plot_paths = []
    episode_labels = planning_episode_labels(planning_stats)
    for method, projections in projections_by_method.items():
        if not projections:
            continue
        for color_key in color_keys:
            values = color_values(metadata, color_key, selection)
            episode_alphas = (
                episode_step_alphas(
                    metadata,
                    selection,
                    max_alpha=args.alpha,
                    min_alpha=args.episode_min_alpha,
                )
                if color_key == "episode_idx"
                else None
            )
            path = output_dir / f"{method}_{color_key}.png"
            plot_projection_grid(
                projections,
                method=method,
                color_key=color_key,
                values=values,
                output_path=path,
                point_size=args.point_size,
                alpha=args.alpha,
                explained_variance=pca_variance if method == "pca" else None,
                episode_colors=args.episode_colors,
                episode_alpha_values=episode_alphas,
                episode_order=selected_episodes,
                episode_labels=episode_labels,
            )
            plot_paths.append(str(path))

    write_projection_csv(
        output_dir / "projection_points.csv",
        projections_by_method,
        metadata,
        selection,
        color_keys,
    )

    summary = {
        "dataset": args.dataset,
        "device": str(device),
        "num_clips": len(indices),
        "selected_episodes": selected_episodes,
        "num_steps": int(prepared[0]["cfg"].data.dataset.num_steps),
        "plot_points": int(len(selection)),
        "standardized_latents": not args.no_standardize,
        "note": (
            "PCA/t-SNE coordinates are fit per model. Compare cluster shape and "
            "metadata organization within each subplot, not absolute x/y coordinates "
            "between independently trained checkpoints."
        ),
        "models": model_summaries,
        "pca_explained_variance_ratio": pca_variance,
        "color_keys": color_keys,
        "plots": plot_paths,
        "projection_csv": "projection_points.csv",
        "planning_prioritization": {
            "log_paths": [str(path) for path in planning_log_paths],
            "model_filters": args.planning_model_filter,
            "outcome": args.planning_outcome,
            "default_goal_offset_steps": args.planning_default_goal_offset_steps,
            "preferred_episodes": planning_preferred_episodes,
            "selected_episode_stats": {
                str(ep): planning_stats[int(ep)]
                for ep in (selected_episodes or [])
                if int(ep) in planning_stats
            },
        },
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nLatent visualizations complete.")
    print(f"Output directory: {output_dir}")
    print(f"Models: {', '.join(latents_by_model)}")
    print(f"Points plotted per model: {len(selection)}")
    print(f"Color keys: {', '.join(color_keys)}")
    print("Saved summary.json, projection_points.csv, and PCA/t-SNE PNGs.")


if __name__ == "__main__":
    main()
