"""Visualize one episode encoded under different missing-modality conditions.

The script encodes the same dataset episode multiple times with a frozen model:
all modalities present, one modality removed, and one modality kept. All
conditions are projected together so their trajectories can be compared in the
same latent coordinate system.
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
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.experiment_utils import (  # noqa: E402
    batch_to_device,
    build_dataset,
    get_dataset_columns,
    load_cost_model,
    load_experiment_config,
)
from experiments.latent_visualizations import (  # noqa: E402
    compute_pca,
    compute_tsne,
    configure_dataset,
    flatten_sequence,
    metadata_keys_to_load,
    sample_episode_clip_indices,
    standardize_latents,
)
from multimodal import get_enabled_modality_configs  # noqa: E402


EPS = 1e-8
MODALITY_COLORS = {
    "pixels": "#5F0FF8",
    "depth": "#F5A12E",
    "tactile": "#008B21",
    "proprio": "#0053E9",
    "force_torque": "#A50025",
}
FULL_COLOR = "#111111"


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    checkpoint: str


@dataclass(frozen=True)
class Condition:
    label: str
    drop_modalities: tuple[str, ...]
    kind: str
    modality: str | None = None


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot a single episode latent trajectory under full, drop-one, and "
            "keep-one modality masking conditions."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint_spec,
        required=True,
        help=(
            "Checkpoint path/run name. Repeat up to three times. "
            "Use LABEL=CHECKPOINT for clean subplot labels."
        ),
    )
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help=(
            "Optional config path. Use PATH for all/single-checkpoint runs, or "
            "LABEL=PATH for per-checkpoint configs."
        ),
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--episode-idx",
        type=int,
        default=None,
        help="Episode id to visualize. Defaults to one sampled episode.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Clip length used for dataset loading. Defaults to checkpoint config.",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max_points",
        type=int,
        default=25000,
        help="Maximum total latent points used for projection across conditions.",
    )
    parser.add_argument(
        "--keep-dataset-cache",
        action="store_true",
        help="Keep keys_to_cache from the checkpoint config.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable latent standardization before PCA/t-SNE.",
    )
    parser.add_argument(
        "--skip-tsne",
        action="store_true",
        help="Only generate the PCA figure.",
    )
    parser.add_argument(
        "--fit-pca-on-reference",
        action="store_true",
        help=(
            "Fit PCA once on many full-modality dataset clips, then project the "
            "diagnostic episode into that fixed PCA space. This makes the full "
            "trajectory directly comparable between drop-one and keep-one plots."
        ),
    )
    parser.add_argument(
        "--reference-num-clips",
        type=int,
        default=1024,
        help="Number of random full-modality clips used to fit reference PCA.",
    )
    parser.add_argument(
        "--reference-max-points",
        type=int,
        default=50000,
        help="Maximum latent points used when fitting reference PCA.",
    )
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-max-iter", type=int, default=1000)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if len(args.checkpoint) > 3:
        parser.error("latent mask visualization accepts at most three checkpoints.")
    labels = [spec.label for spec in args.checkpoint]
    if len(labels) != len(set(labels)):
        parser.error("checkpoint labels must be unique.")

    return args


def parse_config_args(values: list[str]) -> tuple[Path | None, dict[str, Path]]:
    shared = None
    by_label = {}
    for value in values:
        if "=" in value:
            label, path = value.split("=", 1)
            by_label[label.strip()] = Path(path.strip()).expanduser()
        else:
            shared = Path(value).expanduser()
    return shared, by_label


def sanitize_batch(batch: dict) -> dict:
    copied = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            value = value.clone()
            if torch.is_floating_point(value):
                value = torch.nan_to_num(value, 0.0)
        copied[key] = value
    return copied


def model_supports_missing_modalities(model) -> bool:
    encoder = getattr(model, "encoder", None)
    imputer = getattr(encoder, "imputer", None) if encoder is not None else None
    if imputer is not None:
        return bool(getattr(imputer, "supports_missing_modalities", False))
    fusion = getattr(encoder, "fusion", None) if encoder is not None else None
    return bool(getattr(fusion, "supports_missing_modalities", False))


def model_modality_sources(model) -> list[str]:
    encoder = getattr(model, "encoder", None)
    encoders = getattr(encoder, "encoders", None)
    if encoders is None:
        return []

    sources = []
    for modality_encoder in encoders.values():
        source = str(getattr(modality_encoder, "source", ""))
        if source and source not in sources:
            sources.append(source)
    return sources


def config_modality_sources(cfg) -> list[str]:
    if not OmegaConf.select(cfg, "obs_encoder.modalities"):
        return ["pixels"]

    sources = []
    for name, mod_cfg in get_enabled_modality_configs(cfg.obs_encoder).items():
        source = str(mod_cfg.get("source", name))
        if source not in sources:
            sources.append(source)
    return sources


def build_conditions(sources: list[str], supports_missing: bool) -> list[Condition]:
    conditions = [Condition("full", tuple(), "full")]
    if len(sources) <= 1:
        return conditions

    if not supports_missing:
        raise ValueError(
            "This model has multiple modalities but does not advertise missing "
            "modality support. Use a model with an imputer/fusion that supports "
            "missing modalities."
        )

    for source in sources:
        conditions.append(
            Condition(
                label=f"drop_{source}",
                drop_modalities=(source,),
                kind="drop",
                modality=source,
            )
        )

    for source in sources:
        drop = tuple(other for other in sources if other != source)
        if drop:
            conditions.append(
                Condition(
                    label=f"keep_{source}",
                    drop_modalities=drop,
                    kind="keep",
                    modality=source,
                )
            )

    return conditions


def metadata_from_batch(batch: dict, num_steps: int) -> dict[str, np.ndarray]:
    out = {}
    for key in ("episode_idx", "ep_idx", "step_idx"):
        flat = flatten_sequence(batch.get(key), expected_steps=num_steps)
        if flat is None:
            continue
        arr = flat.numpy()
        if arr.ndim > 1 and int(np.prod(arr.shape[1:])) == 1:
            arr = arr.reshape(arr.shape[0])
        out[key] = arr
    return out


def append_metadata(storage: dict[str, list[np.ndarray]], batch: dict, num_steps: int):
    for key, value in metadata_from_batch(batch, num_steps).items():
        storage.setdefault(key, []).append(value)


def finalize_metadata(storage: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate(chunks, axis=0)
        for key, chunks in storage.items()
        if chunks
    }


@torch.no_grad()
def encode_condition(
    *,
    model,
    dataset,
    indices: list[int],
    condition: Condition,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    collect_metadata: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray], int, int]:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    latents = []
    metadata_storage: dict[str, list[np.ndarray]] = {}
    nonfinite_count = 0
    total_values = 0

    for batch in loader:
        clean_batch = sanitize_batch(batch)
        model_batch = batch_to_device(dict(clean_batch), device)
        for modality in condition.drop_modalities:
            model_batch.pop(modality, None)

        output = model.encode(model_batch)
        emb = output["emb"]
        nonfinite_count += int((~torch.isfinite(emb)).sum().item())
        total_values += int(emb.numel())
        emb = torch.nan_to_num(emb, 0.0)
        latents.append(emb.detach().cpu().float().reshape(-1, emb.shape[-1]))

        if collect_metadata:
            append_metadata(metadata_storage, clean_batch, num_steps=int(emb.shape[1]))

    if not latents:
        raise ValueError(f"No latents were extracted for condition {condition.label!r}.")

    metadata = finalize_metadata(metadata_storage) if collect_metadata else {}
    return (
        torch.cat(latents, dim=0).numpy(),
        metadata,
        nonfinite_count,
        total_values,
    )


def select_shared_points(point_count: int, condition_count: int, max_points: int, seed: int):
    if point_count <= 0:
        raise ValueError("No latent points were available for projection.")
    per_condition = max(2, int(max_points) // max(int(condition_count), 1))
    count = min(int(point_count), per_condition)
    if count >= point_count:
        return np.arange(point_count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(point_count, size=count, replace=False))


def sample_reference_indices(
    dataset_len: int,
    count: int,
    seed: int,
    exclude: set[int] | None = None,
) -> list[int]:
    exclude = exclude or set()
    candidates = np.asarray(
        [idx for idx in range(int(dataset_len)) if idx not in exclude],
        dtype=np.int64,
    )
    if candidates.size == 0:
        raise ValueError("No dataset clips available for reference PCA.")
    count = min(int(count), int(candidates.size))
    rng = np.random.default_rng(seed)
    selected = rng.choice(candidates, size=count, replace=False)
    return sorted(int(idx) for idx in selected.tolist())


def split_projection(
    coords: np.ndarray,
    labels: list[str],
    points_per_condition: int,
) -> dict[str, np.ndarray]:
    out = {}
    start = 0
    for label in labels:
        end = start + points_per_condition
        out[label] = coords[start:end]
        start = end
    return out


def average_by_step(
    coords: np.ndarray,
    metadata: dict[str, np.ndarray],
    selection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if "step_idx" not in metadata:
        steps = np.arange(len(coords), dtype=np.int64)
    else:
        steps = np.asarray(metadata["step_idx"])[selection].reshape(-1)

    order = np.argsort(steps)
    steps = steps[order].astype(np.int64)
    coords = coords[order]
    unique_steps = np.unique(steps)

    averaged = []
    counts = []
    for step in unique_steps:
        mask = steps == step
        averaged.append(coords[mask].mean(axis=0))
        counts.append(int(mask.sum()))

    return (
        unique_steps,
        np.asarray(averaged, dtype=np.float32),
        np.asarray(counts, dtype=np.int64),
    )


def condition_style(condition: Condition):
    if condition.kind == "full":
        return FULL_COLOR, "-", 2.8, 1.0

    color = MODALITY_COLORS.get(str(condition.modality), "#777777")
    if condition.kind == "drop":
        return color, "--", 1.8, 0.92
    return color, ":", 2.2, 0.92


def group_conditions(conditions: list[Condition], kind: str) -> list[Condition]:
    selected = [condition for condition in conditions if condition.kind == "full"]
    selected.extend(condition for condition in conditions if condition.kind == kind)
    return selected


def fit_reference_pca(
    latents: np.ndarray,
    *,
    seed: int,
    max_points: int,
    standardize: bool,
) -> dict:
    latents = np.asarray(latents, dtype=np.float32)
    latents = np.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0)
    if latents.shape[0] < 2:
        raise ValueError("Need at least two reference latent points for PCA.")

    if latents.shape[0] > int(max_points):
        rng = np.random.default_rng(seed)
        selection = np.sort(
            rng.choice(latents.shape[0], size=int(max_points), replace=False)
        )
        latents = latents[selection]

    mean = latents.mean(axis=0, keepdims=True)
    std = latents.std(axis=0, keepdims=True)
    if standardize:
        fit_latents = (latents - mean) / np.maximum(std, EPS)
    else:
        fit_latents = latents

    reducer = PCA(n_components=2, random_state=seed)
    reducer.fit(fit_latents)
    return {
        "reducer": reducer,
        "mean": mean,
        "std": std,
        "standardize": bool(standardize),
        "explained_variance": reducer.explained_variance_ratio_.astype(float).tolist(),
        "num_points": int(latents.shape[0]),
    }


def transform_reference_pca(latents: np.ndarray, reference: dict) -> np.ndarray:
    latents = np.asarray(latents, dtype=np.float32)
    latents = np.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0)
    if reference["standardize"]:
        latents = (latents - reference["mean"]) / np.maximum(reference["std"], EPS)
    return reference["reducer"].transform(latents)


def project_conditions(
    *,
    latents_by_condition: dict[str, np.ndarray],
    conditions: list[Condition],
    metadata: dict[str, np.ndarray],
    max_points: int,
    seed: int,
    standardize: bool,
    method: str,
    perplexity: float,
    tsne_max_iter: int,
    pca_reference: dict | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[float] | None]:
    labels = [condition.label for condition in conditions]
    point_count = min(latents_by_condition[label].shape[0] for label in labels)
    selection = select_shared_points(
        point_count,
        condition_count=len(conditions),
        max_points=max_points,
        seed=seed,
    )
    selected_latents = [
        latents_by_condition[label][:point_count][selection] for label in labels
    ]

    if method == "pca" and pca_reference is not None:
        projections = {
            label: transform_reference_pca(latents, pca_reference)
            for label, latents in zip(labels, selected_latents)
        }
        return projections, selection, pca_reference["explained_variance"]

    combined = np.concatenate(selected_latents, axis=0)
    if standardize:
        combined = standardize_latents(combined)
    else:
        combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)

    if method == "pca":
        coords, explained_variance = compute_pca(combined, seed)
        if coords is None:
            raise ValueError("Need at least two latent points for PCA.")
    elif method == "tsne":
        coords = compute_tsne(
            combined,
            perplexity=perplexity,
            max_iter=tsne_max_iter,
            seed=seed,
        )
        if coords is None:
            raise ValueError("Need at least three latent points for t-SNE.")
        explained_variance = None
    else:
        raise ValueError(f"Unknown projection method: {method}")

    return (
        split_projection(coords, labels, points_per_condition=len(selection)),
        selection,
        explained_variance,
    )


def draw_condition_subplot(
    ax,
    *,
    projections: dict[str, np.ndarray],
    conditions: list[Condition],
    metadata: dict[str, np.ndarray],
    selection: np.ndarray,
    method: str,
    model_label: str,
    explained_variance: list[float] | None,
):
    rows = []

    for condition in conditions:
        coords = projections[condition.label]
        steps, step_coords, counts = average_by_step(coords, metadata, selection)
        color, linestyle, linewidth, alpha = condition_style(condition)
        ax.plot(
            step_coords[:, 0],
            step_coords[:, 1],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=condition.label.replace("_", " "),
        )
        ax.scatter(
            step_coords[:, 0],
            step_coords[:, 1],
            s=16 if condition.kind == "full" else 11,
            color=color,
            alpha=alpha,
            linewidths=0,
        )

        for step, coord, count in zip(steps, step_coords, counts):
            rows.append(
                {
                    "model": model_label,
                    "method": method,
                    "condition": condition.label,
                    "kind": condition.kind,
                    "modality": condition.modality or "",
                    "step_idx": int(step),
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "num_points_averaged": int(count),
                }
            )

    title = model_label
    if method == "pca" and explained_variance is not None and len(explained_variance) >= 2:
        total = 100.0 * (float(explained_variance[0]) + float(explained_variance[1]))
        title = f"{model_label}\nPC1+PC2: {total:.1f}%"

    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.grid(True, linewidth=0.3, alpha=0.35)
    return rows


def plot_condition_grid(
    *,
    group_name: str,
    method: str,
    prepared: list[dict],
    output_path: Path,
    dpi: int,
) -> list[dict]:
    ncols = len(prepared)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(5.3 * ncols, 4.7),
        squeeze=False,
    )
    rows = []
    handles = None
    labels = None

    for ax, item in zip(axes.ravel(), prepared):
        subplot_rows = draw_condition_subplot(
            ax,
            projections=item["projections"],
            conditions=item["conditions"],
            metadata=item["metadata"],
            selection=item["selection"],
            method=method,
            model_label=item["label"],
            explained_variance=item["explained_variance"],
        )
        for row in subplot_rows:
            row["group"] = group_name
        rows.extend(subplot_rows)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    if handles and labels:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 4),
            frameon=False,
        )
    fig.suptitle(f"{group_name.replace('_', ' ').title()} | {method.upper()}")
    fig.tight_layout(rect=[0.0, 0.12, 1.0, 0.92])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return rows


def write_projection_csv(path: Path, rows: list[dict]):
    fieldnames = [
        "model",
        "method",
        "group",
        "condition",
        "kind",
        "modality",
        "step_idx",
        "x",
        "y",
        "num_points_averaged",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir.expanduser() if args.cache_dir is not None else None
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    shared_config, config_by_label = parse_config_args(args.config)
    available_columns = get_dataset_columns(args.dataset, cache_dir)

    def load_cfg_for_spec(spec: CheckpointSpec, num_steps: int | None):
        config_path = config_by_label.get(spec.label, shared_config)
        cfg, resolved_config_path = load_experiment_config(
            checkpoint=spec.checkpoint,
            cache_dir=cache_dir,
            config_path=config_path,
            dataset_name=args.dataset,
        )
        configure_dataset(cfg, args.dataset, num_steps, args.keep_dataset_cache)
        return cfg, resolved_config_path

    def build_dataset_for_cfg(cfg):
        extra_metadata_keys, passthrough_keys = metadata_keys_to_load(
            cfg,
            available_columns,
        )
        cfg_sources = config_modality_sources(cfg)
        extra_keys = sorted(
            set(extra_metadata_keys)
            | {source for source in cfg_sources if source in available_columns}
            | ({"action"} if "action" in available_columns else set())
        )
        dataset = build_dataset(
            cfg,
            cache_dir,
            extra_keys_to_load=extra_keys,
            passthrough_keys=passthrough_keys,
        )
        return dataset, cfg_sources

    first_spec = args.checkpoint[0]
    first_cfg, first_config_path = load_cfg_for_spec(first_spec, args.num_steps)
    clip_length = int(first_cfg.data.dataset.num_steps)
    first_dataset, _first_cfg_sources = build_dataset_for_cfg(first_cfg)

    preferred = [args.episode_idx] if args.episode_idx is not None else None
    indices, selected_episodes = sample_episode_clip_indices(
        first_dataset,
        episode_count=1,
        seed=args.seed,
        preferred_episodes=preferred,
    )
    selected_episode = selected_episodes[0]
    print(f"Selected episode: {selected_episode}", flush=True)
    print(f"Selected clips: {len(indices)}", flush=True)

    reference_indices = None
    if args.fit_pca_on_reference:
        reference_indices = sample_reference_indices(
            len(first_dataset),
            args.reference_num_clips,
            seed=args.seed + 1009,
            exclude=set(indices),
        )
        print(
            f"Reference PCA clips: {len(reference_indices)} "
            f"(full-modality encodings)",
            flush=True,
        )

    model_results = []
    for spec_idx, spec in enumerate(args.checkpoint):
        if spec_idx == 0:
            cfg = first_cfg
            config_path = first_config_path
            dataset = first_dataset
            cfg_sources = _first_cfg_sources
        else:
            cfg, config_path = load_cfg_for_spec(spec, clip_length)
            dataset, cfg_sources = build_dataset_for_cfg(cfg)

        print(f"Loading {spec.label}: {spec.checkpoint}", flush=True)
        model = load_cost_model(spec.checkpoint, cache_dir=cache_dir).to(device).eval()
        model.requires_grad_(False)

        sources = model_modality_sources(model) or cfg_sources
        sources = [source for source in sources if source in available_columns]
        if not sources:
            raise ValueError(f"Could not determine modality sources for {spec.label}.")

        supports_missing = model_supports_missing_modalities(model)
        conditions = build_conditions(sources, supports_missing)
        print(
            f"{spec.label} conditions: "
            + ", ".join(condition.label for condition in conditions),
            flush=True,
        )

        pca_reference = None
        reference_summary = None
        if reference_indices is not None:
            model_reference_indices = [
                idx for idx in reference_indices if idx < len(dataset)
            ]
            print(
                f"Fitting reference PCA for {spec.label} from "
                f"{len(model_reference_indices)} clips ...",
                flush=True,
            )
            reference_latents, _metadata, ref_nonfinite_count, ref_total_values = (
                encode_condition(
                    model=model,
                    dataset=dataset,
                    indices=model_reference_indices,
                    condition=Condition("full", tuple(), "full"),
                    device=device,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    collect_metadata=False,
                )
            )
            pca_reference = fit_reference_pca(
                reference_latents,
                seed=args.seed,
                max_points=args.reference_max_points,
                standardize=not args.no_standardize,
            )
            reference_summary = {
                "num_clips": int(len(model_reference_indices)),
                "num_latent_points_before_subsample": int(reference_latents.shape[0]),
                "num_latent_points_used": int(pca_reference["num_points"]),
                "nonfinite_latent_values": int(ref_nonfinite_count),
                "latent_value_count": int(ref_total_values),
                "pca_explained_variance_ratio": pca_reference[
                    "explained_variance"
                ],
            }

        latents_by_condition = {}
        metadata = None
        nonfinite = {}

        for condition_idx, condition in enumerate(conditions):
            print(f"Encoding {spec.label} / {condition.label} ...", flush=True)
            latents, condition_metadata, nonfinite_count, total_values = encode_condition(
                model=model,
                dataset=dataset,
                indices=indices,
                condition=condition,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                collect_metadata=(condition_idx == 0),
            )
            latents_by_condition[condition.label] = latents
            nonfinite[condition.label] = {
                "nonfinite_latent_values": int(nonfinite_count),
                "latent_value_count": int(total_values),
            }
            if metadata is None:
                metadata = condition_metadata

        assert metadata is not None
        model_results.append(
            {
                "label": spec.label,
                "checkpoint": spec.checkpoint,
                "config_path": str(config_path),
                "sources": sources,
                "supports_missing_modalities": bool(supports_missing),
                "conditions": conditions,
                "latents_by_condition": latents_by_condition,
                "metadata": metadata,
                "nonfinite": nonfinite,
                "pca_reference": pca_reference,
                "reference_summary": reference_summary,
                "point_count": int(
                    min(latents.shape[0] for latents in latents_by_condition.values())
                ),
            }
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    projection_rows = []
    projection_summary = {}
    methods = ["pca"] if args.skip_tsne else ["pca", "tsne"]
    groups = {
        "drop_one": "drop",
        "keep_one": "keep",
    }

    for method in methods:
        projection_summary[method] = {}
        for group_name, condition_kind in groups.items():
            prepared_group = []
            projection_summary[method][group_name] = {}

            for result in model_results:
                selected_conditions = group_conditions(
                    result["conditions"],
                    condition_kind,
                )
                if len(selected_conditions) <= 1:
                    continue
                projections, selection, explained_variance = project_conditions(
                    latents_by_condition=result["latents_by_condition"],
                    conditions=selected_conditions,
                    metadata=result["metadata"],
                    max_points=args.max_points,
                    seed=args.seed,
                    standardize=not args.no_standardize,
                    method=method,
                    perplexity=args.tsne_perplexity,
                    tsne_max_iter=args.tsne_max_iter,
                    pca_reference=(
                        result["pca_reference"] if method == "pca" else None
                    ),
                )
                prepared_group.append(
                    {
                        "label": result["label"],
                        "conditions": selected_conditions,
                        "metadata": result["metadata"],
                        "selection": selection,
                        "projections": projections,
                        "explained_variance": explained_variance,
                    }
                )
                projection_summary[method][group_name][result["label"]] = {
                    "conditions": [condition.label for condition in selected_conditions],
                    "points_per_condition_projected": int(len(selection)),
                    "pca_explained_variance_ratio": explained_variance
                    if method == "pca"
                    else None,
                }

            if not prepared_group:
                continue

            for suffix in ("png", "pdf"):
                rows = plot_condition_grid(
                    group_name=group_name,
                    method=method,
                    prepared=prepared_group,
                    output_path=output_dir
                    / f"latent_mask_{group_name}_{method}.{suffix}",
                    dpi=args.dpi,
                )
                if suffix == "png":
                    projection_rows.extend(rows)

    write_projection_csv(output_dir / "projection_points.csv", projection_rows)

    summary = {
        "checkpoints": [
            {
                "label": result["label"],
                "checkpoint": result["checkpoint"],
                "config_path": result["config_path"],
                "sources": result["sources"],
                "supports_missing_modalities": result[
                    "supports_missing_modalities"
                ],
                "conditions": [
                    {
                        "label": condition.label,
                        "kind": condition.kind,
                        "modality": condition.modality,
                        "drop_modalities": list(condition.drop_modalities),
                    }
                    for condition in result["conditions"]
                ],
                "points_per_condition_before_subsample": result["point_count"],
                "reference_pca": result["reference_summary"],
                "nonfinite": result["nonfinite"],
            }
            for result in model_results
        ],
        "dataset": args.dataset,
        "selected_episode": int(selected_episode),
        "selected_clip_count": int(len(indices)),
        "clip_length": int(clip_length),
        "standardized": not args.no_standardize,
        "pca_basis": (
            "reference_full_modalities"
            if args.fit_pca_on_reference
            else "plotted_conditions"
        ),
        "projection_summary": projection_summary,
        "outputs": {
            "drop_one_pca_png": "latent_mask_drop_one_pca.png",
            "drop_one_pca_pdf": "latent_mask_drop_one_pca.pdf",
            "keep_one_pca_png": "latent_mask_keep_one_pca.png",
            "keep_one_pca_pdf": "latent_mask_keep_one_pca.pdf",
            "drop_one_tsne_png": None
            if args.skip_tsne
            else "latent_mask_drop_one_tsne.png",
            "drop_one_tsne_pdf": None
            if args.skip_tsne
            else "latent_mask_drop_one_tsne.pdf",
            "keep_one_tsne_png": None
            if args.skip_tsne
            else "latent_mask_keep_one_tsne.png",
            "keep_one_tsne_pdf": None
            if args.skip_tsne
            else "latent_mask_keep_one_tsne.pdf",
            "csv": "projection_points.csv",
        }
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved visualizations to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
