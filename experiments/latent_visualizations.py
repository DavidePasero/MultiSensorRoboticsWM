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
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    "step_idx",
    "ee_object_distance",
    "ee_target_distance",
    "bool_contact",
    "success",
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
    return parser.parse_args()


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
    finite = np.isfinite(values.astype(np.float64, copy=False))
    binary = is_binary(values)
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

    for ax in axes_flat[len(labels) :]:
        ax.axis("off")

    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes_flat[: len(labels)], shrink=0.82)
        cbar.set_label(color_key)
        if binary:
            cbar.set_ticks([0, 1])

    fig.suptitle(f"{method.upper()} latent visualization colored by {color_key}")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
    if len({spec.label for spec in args.checkpoint}) != len(args.checkpoint):
        raise ValueError("Checkpoint labels must be unique.")

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
    for method, projections in projections_by_method.items():
        if not projections:
            continue
        for color_key in color_keys:
            values = color_values(metadata, color_key, selection)
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
