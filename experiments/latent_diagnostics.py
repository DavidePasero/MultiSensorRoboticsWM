"""Latent-space diagnostics for frozen world-model checkpoints.

The script samples contiguous dataset clips, encodes them with a checkpoint, and
reports trajectory smoothness, action sensitivity, open-loop prediction error,
and latent nearest-neighbour sanity checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from experiments.experiment_utils import (  # noqa: E402
    build_dataset,
    get_dataset_columns,
    load_cost_model,
    load_experiment_config,
)
from multimodal import get_enabled_modality_configs  # noqa: E402

EPS = 1e-8
PHYSICAL_KEYS = ("ee_xyz", "object_1_xyz", "object_2_xyz", "proprio")
IMAGE_KEYS = ("pixels", "rgb", "image", "depth", "tactile")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run latent-space diagnostics on a frozen LeWM checkpoint.",
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path or run name.")
    parser.add_argument("--dataset", required=True, help="Dataset name in STABLEWM cache.")
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None, help="Optional config.yaml path.")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument("--num_trajectories", type=int, default=128)
    parser.add_argument("--max_horizon", type=int, default=10)
    parser.add_argument("--num_nn_queries", type=int, default=8)
    parser.add_argument("--num_neighbors", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--image_key",
        default="auto",
        help="Dataset key used for nearest-neighbour visualizations.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for sampled clips.",
    )
    parser.add_argument(
        "--keys-to-cache",
        nargs="*",
        default=None,
        help=(
            "Override data.dataset.keys_to_cache. Use no values to disable "
            "RAM caching, e.g. --keys-to-cache."
        ),
    )
    parser.add_argument(
        "--cache-all-loaded",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override data.dataset.cache_all_loaded when the dataset config has it.",
    )
    parser.add_argument(
        "--mask-modalities",
        nargs="+",
        default=None,
        help=(
            "Modalities to drop for full-vs-masked rollout divergence, e.g. "
            "--mask-modalities pixels depth."
        ),
    )
    parser.add_argument(
        "--keep-modalities",
        nargs="+",
        default=None,
        help=(
            "Keep only these modalities for full-vs-masked rollout divergence; "
            "all other enabled observation modalities are dropped."
        ),
    )
    parser.add_argument(
        "--mask-each-modality",
        action="store_true",
        help="Also evaluate one full-vs-masked divergence condition per modality.",
    )
    return parser.parse_args()


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(val, device) for key, val in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    return value


def clone_model_batch(batch):
    copied = {}
    for key, value in batch.items():
        copied[key] = value.clone() if torch.is_tensor(value) else value
    if "action" in copied:
        copied["action"] = torch.nan_to_num(copied["action"], 0.0)
    return copied


def tensor_to_cpu(value):
    if not torch.is_tensor(value):
        return None
    return value.detach().cpu()


def infer_history_size(model, cfg) -> int:
    predictor = getattr(model, "predictor", None)
    pos_embedding = getattr(predictor, "pos_embedding", None)
    if torch.is_tensor(pos_embedding):
        return int(pos_embedding.shape[1])
    return int(OmegaConf.select(cfg, "wm.history_size", default=1))


def enabled_model_sources(cfg) -> set[str]:
    if not OmegaConf.select(cfg, "obs_encoder.modalities"):
        return {"pixels", "action"}
    sources = {"action"}
    for name, mod_cfg in get_enabled_modality_configs(cfg.obs_encoder).items():
        sources.add(str(mod_cfg.get("source", name)))
    return sources


def enabled_observation_sources(cfg) -> list[str]:
    if not OmegaConf.select(cfg, "obs_encoder.modalities"):
        return ["pixels"]
    sources = []
    for name, mod_cfg in get_enabled_modality_configs(cfg.obs_encoder).items():
        source = str(mod_cfg.get("source", name))
        if source not in sources:
            sources.append(source)
    return sources


def model_supports_missing_modalities(model) -> bool:
    encoder = getattr(model, "encoder", None)
    imputer = getattr(encoder, "imputer", None) if encoder is not None else None
    if imputer is not None:
        return bool(getattr(imputer, "supports_missing_modalities", False))
    fusion = getattr(encoder, "fusion", None) if encoder is not None else None
    return bool(getattr(fusion, "supports_missing_modalities", False))


def build_mask_conditions(cfg, args) -> list[tuple[str, list[str]]]:
    sources = enabled_observation_sources(cfg)
    conditions = []

    def add_condition(label: str, drop_modalities: list[str]):
        unknown = [name for name in drop_modalities if name not in sources]
        if unknown:
            raise ValueError(
                f"Unknown mask modalities {unknown}. Available modalities: {sources}."
            )
        drop_modalities = [name for name in sources if name in drop_modalities]
        if not drop_modalities:
            return
        if len(drop_modalities) >= len(sources):
            raise ValueError(
                f"Mask condition '{label}' drops every modality. "
                f"Available modalities: {sources}."
            )
        condition = (label, drop_modalities)
        if condition not in conditions:
            conditions.append(condition)

    if args.mask_each_modality:
        for source in sources:
            if len(sources) > 1:
                add_condition(f"drop_{source}", [source])

    if args.mask_modalities is not None:
        label = "drop_" + "_".join(args.mask_modalities)
        add_condition(label, list(args.mask_modalities))

    if args.keep_modalities is not None:
        unknown = [name for name in args.keep_modalities if name not in sources]
        if unknown:
            raise ValueError(
                f"Unknown keep modalities {unknown}. Available modalities: {sources}."
            )
        drop = [source for source in sources if source not in args.keep_modalities]
        label = "keep_" + "_".join(args.keep_modalities)
        add_condition(label, drop)

    return conditions


def choose_image_key(requested_key: str, available_columns: list[str]) -> str | None:
    if requested_key != "auto":
        return requested_key if requested_key in available_columns else None
    for key in IMAGE_KEYS:
        if key in available_columns:
            return key
    return None


def configure_dataset(cfg, args, history_size: int):
    clip_len = max(2, history_size + int(args.max_horizon))
    with open_dict(cfg):
        cfg.data.dataset.name = args.dataset
        cfg.data.dataset.num_steps = clip_len
        if args.keys_to_cache is not None:
            cfg.data.dataset.keys_to_cache = list(args.keys_to_cache)
        if args.cache_all_loaded is not None:
            cfg.data.dataset.cache_all_loaded = bool(args.cache_all_loaded)


def build_diagnostic_dataset(cfg, args, cache_dir, available_columns):
    model_sources = enabled_model_sources(cfg)
    image_key = choose_image_key(args.image_key, available_columns)
    physical_keys = [
        key
        for key in PHYSICAL_KEYS
        if key in available_columns and key not in model_sources
    ]

    required_keys = []
    for key in [image_key, *physical_keys]:
        if key is not None and key not in cfg.data.dataset.keys_to_load:
            required_keys.append(key)

    passthrough_keys = [
        key
        for key in [image_key, *physical_keys]
        if key is not None and key not in model_sources
    ]

    dataset = build_dataset(
        cfg,
        cache_dir,
        extra_keys_to_load=required_keys,
        passthrough_keys=passthrough_keys,
    )
    return dataset, image_key, physical_keys


def sample_indices(dataset_len: int, count: int, seed: int) -> list[int]:
    if dataset_len <= 0:
        raise ValueError("Dataset produced no valid clips for diagnostics.")
    count = min(int(count), dataset_len)
    rng = np.random.default_rng(seed)
    return rng.choice(dataset_len, size=count, replace=False).tolist()


def make_loader(dataset, indices, batch_size: int, num_workers: int, pin_memory: bool):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


@torch.no_grad()
def encode_batch(model, batch):
    output = model.encode(clone_model_batch(batch))
    emb = output["emb"]
    if "action" not in batch:
        raise KeyError("Latent diagnostics require an 'action' key in the dataset.")
    action = torch.nan_to_num(batch["action"], 0.0)
    return emb, action


def mask_batch(batch, drop_modalities: list[str]):
    masked = dict(batch)
    for modality in drop_modalities:
        masked.pop(modality, None)
    return masked


@torch.no_grad()
def open_loop_predictions(model, emb, action, history_size: int, max_horizon: int):
    max_valid_horizon = min(
        int(max_horizon),
        emb.size(1) - history_size,
        max(0, action.size(1) - history_size + 1),
    )
    predictions = {}
    if max_valid_horizon <= 0:
        return predictions

    rollout_emb = emb[:, :history_size].clone()
    rollout_action = action[:, :history_size].clone()

    for horizon in range(1, max_valid_horizon + 1):
        act_emb = model.action_encoder(rollout_action[:, -history_size:])
        pred_next = model.predict(
            rollout_emb[:, -history_size:],
            act_emb,
        )[:, -1]
        predictions[horizon] = pred_next.detach()

        if horizon < max_valid_horizon:
            rollout_emb = torch.cat([rollout_emb, pred_next.unsqueeze(1)], dim=1)
            next_action = action[:, history_size + horizon - 1 : history_size + horizon]
            rollout_action = torch.cat([rollout_action, next_action], dim=1)

    return predictions


@torch.no_grad()
def open_loop_errors(model, emb, action, history_size: int, max_horizon: int):
    predictions = open_loop_predictions(
        model,
        emb,
        action,
        history_size,
        max_horizon,
    )
    errors = {horizon: [] for horizon in predictions}
    for horizon, pred_next in predictions.items():
        target = emb[:, history_size + horizon - 1]
        error = torch.linalg.norm(pred_next - target, dim=-1)
        errors[horizon].extend(error.detach().cpu().tolist())

    return errors


def latent_cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    numerator = (a * b).sum(dim=-1)
    denominator = torch.linalg.norm(a, dim=-1) * torch.linalg.norm(b, dim=-1)
    return numerator / denominator.clamp_min(EPS)


def update_encoded_divergence(storage, condition, full_emb, masked_emb):
    steps = min(full_emb.size(1), masked_emb.size(1))
    if steps <= 0:
        return
    full_emb = full_emb[:, :steps]
    masked_emb = masked_emb[:, :steps]
    l2 = torch.linalg.norm(full_emb - masked_emb, dim=-1)
    cosine = latent_cosine(full_emb, masked_emb)
    storage[condition]["l2"].extend(l2.detach().cpu().reshape(-1).tolist())
    storage[condition]["cosine"].extend(cosine.detach().cpu().reshape(-1).tolist())


def update_rollout_divergence(storage, condition, full_predictions, masked_predictions):
    for horizon in sorted(set(full_predictions) & set(masked_predictions)):
        full_pred = full_predictions[horizon]
        masked_pred = masked_predictions[horizon]
        l2 = torch.linalg.norm(full_pred - masked_pred, dim=-1)
        cosine = latent_cosine(full_pred, masked_pred)
        storage[condition]["l2"].setdefault(horizon, []).extend(
            l2.detach().cpu().tolist()
        )
        storage[condition]["cosine"].setdefault(horizon, []).extend(
            cosine.detach().cpu().tolist()
        )


def update_path_metrics(metrics, emb_cpu):
    deltas = emb_cpu[:, 1:] - emb_cpu[:, :-1]
    step_dist = torch.linalg.norm(deltas, dim=-1)
    path_length = step_dist.sum(dim=1)
    endpoint = torch.linalg.norm(emb_cpu[:, -1] - emb_cpu[:, 0], dim=-1)
    tortuosity = path_length / (endpoint + EPS)

    metrics["step_distance_mean"].extend(step_dist.mean(dim=1).tolist())
    metrics["path_length"].extend(path_length.tolist())
    metrics["endpoint_distance"].extend(endpoint.tolist())
    metrics["tortuosity"].extend(tortuosity.tolist())


def update_action_metrics(metrics, emb_cpu, action_cpu, global_pairs):
    latent_delta = torch.linalg.norm(emb_cpu[:, 1:] - emb_cpu[:, :-1], dim=-1)
    action_flat = action_cpu.reshape(action_cpu.size(0), action_cpu.size(1), -1)
    action_norm = torch.linalg.norm(action_flat, dim=-1)
    steps = min(latent_delta.size(1), action_norm.size(1))
    if steps <= 0:
        return

    latent_delta = latent_delta[:, :steps]
    action_norm = action_norm[:, :steps]
    sensitivity = latent_delta / (action_norm + EPS)

    metrics["sensitivity"].extend(sensitivity.mean(dim=1).tolist())
    metrics["latent_delta_mean"].extend(latent_delta.mean(dim=1).tolist())
    metrics["action_norm_mean"].extend(action_norm.mean(dim=1).tolist())
    global_pairs["latent_delta"].append(latent_delta.reshape(-1).numpy())
    global_pairs["action_norm"].append(action_norm.reshape(-1).numpy())


def flatten_sequence_tensor(value):
    if value is None or not torch.is_tensor(value):
        return None
    if value.ndim < 2:
        return None
    return value.reshape(value.shape[0] * value.shape[1], *value.shape[2:])


def append_nn_candidates(storage, emb_cpu, batch_cpu, image_key, physical_keys):
    storage["latents"].append(flatten_sequence_tensor(emb_cpu).float())

    if image_key is not None and image_key in batch_cpu:
        images = flatten_sequence_tensor(batch_cpu[image_key])
        if images is not None:
            storage["images"].append(images)

    physical_parts = []
    for key in physical_keys:
        value = batch_cpu.get(key)
        if not torch.is_tensor(value):
            continue
        flat = flatten_sequence_tensor(value.float())
        if flat is not None:
            physical_parts.append(flat.reshape(flat.size(0), -1))
    if physical_parts:
        storage["physical"].append(torch.cat(physical_parts, dim=-1))


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "count": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "count": int(arr.size),
    }


def pearson_correlation(x_chunks, y_chunks):
    if not x_chunks or not y_chunks:
        return None
    x = np.concatenate(x_chunks).astype(np.float64)
    y = np.concatenate(y_chunks).astype(np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2 or np.std(x) <= EPS or np.std(y) <= EPS:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def prediction_error_rows(errors_by_horizon):
    rows = []
    for horizon in sorted(errors_by_horizon):
        stats = summarize(errors_by_horizon[horizon])
        rows.append(
            {
                "horizon": horizon,
                "mean_error": stats["mean"],
                "std_error": stats["std"],
            }
        )
    return rows


def write_prediction_csv(path: Path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["horizon", "mean_error", "std_error"])
        writer.writeheader()
        writer.writerows(rows)


def encoded_divergence_rows(encoded_divergence):
    rows = []
    for condition in sorted(encoded_divergence):
        l2_stats = summarize(encoded_divergence[condition]["l2"])
        cosine_stats = summarize(encoded_divergence[condition]["cosine"])
        rows.append(
            {
                "condition": condition,
                "mean_l2": l2_stats["mean"],
                "std_l2": l2_stats["std"],
                "mean_cosine": cosine_stats["mean"],
                "std_cosine": cosine_stats["std"],
                "count": l2_stats["count"],
            }
        )
    return rows


def rollout_divergence_rows(rollout_divergence):
    rows = []
    for condition in sorted(rollout_divergence):
        horizons = sorted(
            set(rollout_divergence[condition]["l2"])
            | set(rollout_divergence[condition]["cosine"])
        )
        for horizon in horizons:
            l2_stats = summarize(rollout_divergence[condition]["l2"].get(horizon, []))
            cosine_stats = summarize(
                rollout_divergence[condition]["cosine"].get(horizon, [])
            )
            rows.append(
                {
                    "condition": condition,
                    "horizon": horizon,
                    "mean_l2": l2_stats["mean"],
                    "std_l2": l2_stats["std"],
                    "mean_cosine": cosine_stats["mean"],
                    "std_cosine": cosine_stats["std"],
                    "count": l2_stats["count"],
                }
            )
    return rows


def write_encoded_divergence_csv(path: Path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "mean_l2",
                "std_l2",
                "mean_cosine",
                "std_cosine",
                "count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_rollout_divergence_csv(path: Path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "horizon",
                "mean_l2",
                "std_l2",
                "mean_cosine",
                "std_cosine",
                "count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def as_uint8_image(tensor):
    arr = tensor.detach().cpu().numpy()
    arr = np.asarray(arr)
    input_is_float = np.issubdtype(arr.dtype, np.floating)

    if arr.ndim == 3 and arr.shape[0] in (1, 2, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        arr = arr.reshape(arr.shape[-2], arr.shape[-1], 1)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 2:
        left = np.repeat(arr[..., :1], 3, axis=-1)
        right = np.repeat(arr[..., 1:2], 3, axis=-1)
        arr = np.concatenate([left, right], axis=1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]

    arr = arr.astype(np.float32)
    if np.nanmax(arr) <= 1.5 and np.nanmin(arr) >= -0.1:
        arr = arr * 255.0
    else:
        finite = arr[np.isfinite(arr)]
        if finite.size and (
            input_is_float or finite.max() > 255.0 or finite.min() < 0.0
        ):
            low, high = np.percentile(finite, [1, 99])
            arr = (arr - low) / max(high - low, EPS) * 255.0

    arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(arr, 0, 255).astype(np.uint8)


def make_labeled_tile(image, label: str, tile_size: int = 128):
    image = cv2.resize(image, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    label_bar = np.full((28, tile_size, 3), 255, dtype=np.uint8)
    cv2.putText(
        label_bar,
        label[:24],
        (4, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return np.concatenate([label_bar, image], axis=0)


def save_nearest_neighbor_visuals(
    output_dir: Path,
    latents: torch.Tensor,
    images: torch.Tensor | None,
    physical: torch.Tensor | None,
    *,
    num_queries: int,
    num_neighbors: int,
    seed: int,
):
    if images is None or latents.numel() == 0:
        return []

    nn_dir = output_dir / "nearest_neighbors"
    nn_dir.mkdir(parents=True, exist_ok=True)

    latents = latents.float()
    query_count = min(num_queries, latents.size(0))
    neighbor_count = min(num_neighbors, max(0, latents.size(0) - 1))
    rng = np.random.default_rng(seed)
    query_indices = rng.choice(latents.size(0), size=query_count, replace=False)
    records = []

    for query_id, query_idx in enumerate(query_indices):
        query = latents[int(query_idx)]
        distances = torch.linalg.norm(latents - query.unsqueeze(0), dim=-1)
        distances[int(query_idx)] = math.inf
        nn_indices = torch.topk(
            distances,
            k=neighbor_count,
            largest=False,
        ).indices.tolist()

        tiles = [make_labeled_tile(as_uint8_image(images[int(query_idx)]), "query")]
        for rank, nn_idx in enumerate(nn_indices, start=1):
            latent_distance = float(distances[int(nn_idx)].item())
            physical_distance = None
            if physical is not None:
                physical_distance = float(
                    torch.linalg.norm(
                        physical[int(query_idx)].float() - physical[int(nn_idx)].float()
                    ).item()
                )
            label = f"nn{rank} z={latent_distance:.2f}"
            if physical_distance is not None:
                label += f" p={physical_distance:.2f}"
            tiles.append(make_labeled_tile(as_uint8_image(images[int(nn_idx)]), label))

            records.append(
                {
                    "query_id": query_id,
                    "query_flat_index": int(query_idx),
                    "rank": rank,
                    "neighbor_flat_index": int(nn_idx),
                    "latent_distance": latent_distance,
                    "physical_distance": physical_distance,
                    "image_path": str(nn_dir / f"nn_query_{query_id:03d}.png"),
                }
            )

        grid = np.concatenate(tiles, axis=1)
        cv2.imwrite(str(nn_dir / f"nn_query_{query_id:03d}.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    return records


def write_nn_csv(path: Path, records):
    fieldnames = [
        "query_id",
        "query_flat_index",
        "rank",
        "neighbor_flat_index",
        "latent_distance",
        "physical_distance",
        "image_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def print_summary(summary, prediction_rows, output_dir):
    print("\nLatent diagnostics complete.")
    print(f"Output directory: {output_dir}")
    print(f"Sampled clips: {summary['num_trajectories']}")
    print(
        "Path length: "
        f"{summary['latent_path_length']['path_length']['mean']:.4f} +/- "
        f"{summary['latent_path_length']['path_length']['std']:.4f}"
    )
    print(
        "Tortuosity: "
        f"{summary['latent_path_length']['tortuosity']['mean']:.4f} +/- "
        f"{summary['latent_path_length']['tortuosity']['std']:.4f}"
    )
    print(
        "Action sensitivity: "
        f"{summary['action_sensitivity']['sensitivity']['mean']:.4f} +/- "
        f"{summary['action_sensitivity']['sensitivity']['std']:.4f}"
    )
    if prediction_rows:
        last = prediction_rows[-1]
        print(
            f"Open-loop error at horizon {last['horizon']}: "
            f"{last['mean_error']:.4f} +/- {last['std_error']:.4f}"
        )
    corr = summary["action_sensitivity"].get("global_action_delta_correlation")
    if corr is not None:
        print(f"Global action/latent-delta correlation: {corr:.4f}")
    conditions = summary.get("full_vs_masked_rollout_divergence", {}).get(
        "conditions",
        [],
    )
    if conditions:
        print("Full-vs-masked rollout divergence conditions: " + ", ".join(conditions))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cache_dir = args.cache_dir.expanduser() if args.cache_dir is not None else None
    cfg, config_path = load_experiment_config(
        checkpoint=args.checkpoint,
        cache_dir=cache_dir,
        config_path=args.config,
        dataset_name=args.dataset,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_cost_model(args.checkpoint, cache_dir=cache_dir).to(device).eval()
    model.requires_grad_(False)
    history_size = infer_history_size(model, cfg)
    configure_dataset(cfg, args, history_size)
    mask_conditions = build_mask_conditions(cfg, args)
    if mask_conditions and not model_supports_missing_modalities(model):
        raise ValueError(
            "Full-vs-masked diagnostics require a model whose imputer/fusion "
            "supports missing modalities."
        )

    available_columns = get_dataset_columns(args.dataset, cache_dir)
    dataset, image_key, physical_keys = build_diagnostic_dataset(
        cfg,
        args,
        cache_dir,
        available_columns,
    )

    indices = sample_indices(len(dataset), args.num_trajectories, args.seed)
    loader = make_loader(
        dataset,
        indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    path_metrics = {
        "step_distance_mean": [],
        "path_length": [],
        "endpoint_distance": [],
        "tortuosity": [],
    }
    action_metrics = {
        "sensitivity": [],
        "latent_delta_mean": [],
        "action_norm_mean": [],
    }
    global_pairs = {"latent_delta": [], "action_norm": []}
    prediction_errors = {h: [] for h in range(1, args.max_horizon + 1)}
    encoded_divergence = {
        condition: {"l2": [], "cosine": []} for condition, _ in mask_conditions
    }
    rollout_divergence = {
        condition: {"l2": {}, "cosine": {}} for condition, _ in mask_conditions
    }
    nn_storage = {"latents": [], "images": [], "physical": []}

    with torch.no_grad():
        for batch in loader:
            batch_cpu = {key: tensor_to_cpu(value) for key, value in batch.items()}
            batch_device = to_device(batch, device)
            emb, action = encode_batch(model, batch_device)

            update_path_metrics(path_metrics, emb.detach().cpu())
            update_action_metrics(
                action_metrics,
                emb.detach().cpu(),
                action.detach().cpu(),
                global_pairs,
            )
            for horizon, values in open_loop_errors(
                model,
                emb,
                action,
                history_size,
                args.max_horizon,
            ).items():
                prediction_errors[horizon].extend(values)

            if mask_conditions:
                full_predictions = open_loop_predictions(
                    model,
                    emb,
                    action,
                    history_size,
                    args.max_horizon,
                )
                for condition, drop_modalities in mask_conditions:
                    masked_emb, _masked_action = encode_batch(
                        model,
                        mask_batch(batch_device, drop_modalities),
                    )
                    update_encoded_divergence(
                        encoded_divergence,
                        condition,
                        emb,
                        masked_emb,
                    )
                    masked_predictions = open_loop_predictions(
                        model,
                        masked_emb,
                        action,
                        history_size,
                        args.max_horizon,
                    )
                    update_rollout_divergence(
                        rollout_divergence,
                        condition,
                        full_predictions,
                        masked_predictions,
                    )

            append_nn_candidates(
                nn_storage,
                emb.detach().cpu(),
                batch_cpu,
                image_key,
                physical_keys,
            )

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = prediction_error_rows(prediction_errors)
    write_prediction_csv(output_dir / "prediction_error.csv", prediction_rows)
    encoded_rows = encoded_divergence_rows(encoded_divergence)
    rollout_rows = rollout_divergence_rows(rollout_divergence)
    write_encoded_divergence_csv(
        output_dir / "full_vs_masked_encoded_divergence.csv",
        encoded_rows,
    )
    write_rollout_divergence_csv(
        output_dir / "full_vs_masked_rollout_divergence.csv",
        rollout_rows,
    )

    latents = torch.cat(nn_storage["latents"], dim=0) if nn_storage["latents"] else None
    images = torch.cat(nn_storage["images"], dim=0) if nn_storage["images"] else None
    physical = (
        torch.cat(nn_storage["physical"], dim=0) if nn_storage["physical"] else None
    )
    nn_records = []
    if latents is not None:
        nn_records = save_nearest_neighbor_visuals(
            output_dir,
            latents,
            images,
            physical,
            num_queries=args.num_nn_queries,
            num_neighbors=args.num_neighbors,
            seed=args.seed,
        )
    write_nn_csv(output_dir / "nearest_neighbors.csv", nn_records)

    summary = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "config_path": str(config_path),
        "num_trajectories": len(indices),
        "clip_length": int(cfg.data.dataset.num_steps),
        "history_size": history_size,
        "max_horizon": int(args.max_horizon),
        "device": str(device),
        "image_key": image_key,
        "physical_keys": physical_keys,
        "latent_path_length": {
            key: summarize(values) for key, values in path_metrics.items()
        },
        "action_sensitivity": {
            key: summarize(values) for key, values in action_metrics.items()
        },
        "prediction_error_csv": "prediction_error.csv",
        "full_vs_masked_encoded_divergence_csv": (
            "full_vs_masked_encoded_divergence.csv"
        ),
        "full_vs_masked_rollout_divergence_csv": (
            "full_vs_masked_rollout_divergence.csv"
        ),
        "full_vs_masked_encoded_divergence": {
            row["condition"]: {
                key: value
                for key, value in row.items()
                if key != "condition"
            }
            for row in encoded_rows
        },
        "full_vs_masked_rollout_divergence": {
            "conditions": [condition for condition, _ in mask_conditions],
            "drop_modalities": {
                condition: drop_modalities
                for condition, drop_modalities in mask_conditions
            },
            "by_condition_horizon": {
                row["condition"]: {
                    **{
                        str(other["horizon"]): {
                            key: value
                            for key, value in other.items()
                            if key not in {"condition", "horizon"}
                        }
                        for other in rollout_rows
                        if other["condition"] == row["condition"]
                    }
                }
                for row in rollout_rows
            },
        },
        "nearest_neighbor_csv": "nearest_neighbors.csv",
        "nearest_neighbor_images": len(
            sorted((output_dir / "nearest_neighbors").glob("*.png"))
        )
        if (output_dir / "nearest_neighbors").exists()
        else 0,
    }
    summary["action_sensitivity"]["global_action_delta_correlation"] = pearson_correlation(
        global_pairs["action_norm"],
        global_pairs["latent_delta"],
    )

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary, prediction_rows, output_dir)


if __name__ == "__main__":
    main()
