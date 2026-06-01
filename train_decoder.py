"""Train diagnostic decoders on top of a frozen LeWM checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decoder import (
    BINARY_TARGETS,
    build_decoder_bank,
    decoder_loss_and_metrics,
    infer_target_specs,
    prepare_decoder_target,
    scalar_metrics_to_float,
)
from experiments.experiment_utils import (
    batch_to_device,
    build_dataset,
    get_dataset_columns,
    load_cost_model,
    load_experiment_config,
)
from multimodal import get_enabled_modality_configs

DEFAULT_DECODER_CONFIG_PATH = REPO_ROOT / "config" / "decoder" / "train_decoder.yaml"


def _resolve_optional_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    path = Path(path).expanduser()
    if path.is_absolute() or path.exists():
        return path
    repo_path = REPO_ROOT / path
    return repo_path if repo_path.exists() else path


def load_decoder_training_config(config_path: Path | None):
    resolved_path = _resolve_optional_path(config_path)
    if resolved_path is None or not resolved_path.exists():
        return OmegaConf.create({}), resolved_path
    return OmegaConf.load(resolved_path), resolved_path


def _cfg_get(cfg, key: str, default=None):
    return OmegaConf.select(cfg, key, default=default)


def _plain_config_value(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _cfg_list(value):
    value = _plain_config_value(value)
    if value is None:
        return None
    if isinstance(value, str):
        return value.split()
    return list(value)


def _cfg_loss_weights(value):
    value = _plain_config_value(value)
    if value is None:
        return []
    if isinstance(value, dict):
        return [f"{key}={val}" for key, val in value.items()]
    return list(value)


def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--decoder-config",
        type=Path,
        default=DEFAULT_DECODER_CONFIG_PATH,
        help="YAML config for decoder training defaults and WandB logging.",
    )
    pre_args, _ = pre_parser.parse_known_args()
    decoder_cfg, decoder_config_path = load_decoder_training_config(
        pre_args.decoder_config,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Train post-hoc decoders from frozen LeWM latents and evaluate "
            "encoded-state versus predicted-future reconstruction."
        ),
        parents=[pre_parser],
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=_cfg_get(decoder_cfg, "checkpoint"),
        help="Checkpoint path or run reference.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_cfg_get(decoder_cfg, "model_config"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_cfg_get(decoder_cfg, "cache_dir"),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=_cfg_get(decoder_cfg, "dataset_name"),
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=_cfg_list(_cfg_get(decoder_cfg, "targets")),
        help=(
            "Dataset keys to decode. Defaults to enabled observation modalities "
            "plus bool_contact when present."
        ),
    )
    parser.add_argument(
        "--loss-weight",
        action="append",
        default=_cfg_loss_weights(_cfg_get(decoder_cfg, "loss_weights")),
        metavar="KEY=VALUE",
        help="Optional per-target loss weight. Can be passed multiple times.",
    )
    parser.add_argument(
        "--train-on",
        choices=("all", "future"),
        default=_cfg_get(decoder_cfg, "training.train_on", "all"),
        help="Train decoders on all encoded clip steps or only predictor target steps.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=_cfg_get(decoder_cfg, "training.max_samples", 50000),
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=_cfg_get(decoder_cfg, "training.train_fraction", 0.7),
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=_cfg_get(decoder_cfg, "training.val_fraction", 0.15),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_cfg_get(decoder_cfg, "loader.batch_size", 32),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=_cfg_get(decoder_cfg, "loader.num_workers", 0),
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=_cfg_get(decoder_cfg, "trainer.num_epochs", 50),
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=_cfg_get(decoder_cfg, "optimizer.lr", 1e-3),
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=_cfg_get(decoder_cfg, "optimizer.weight_decay", 1e-4),
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=_cfg_get(decoder_cfg, "trainer.patience", 10),
    )
    parser.add_argument("--seed", type=int, default=_cfg_get(decoder_cfg, "seed", 0))
    parser.add_argument(
        "--device",
        type=str,
        default=_cfg_get(decoder_cfg, "device"),
    )
    parser.add_argument(
        "--pixel-image-size",
        type=int,
        default=_cfg_get(decoder_cfg, "pixel_decoder.image_size", 224),
        help="Pixel decoder output size. Use 0 to decode to dataset-native size.",
    )
    parser.add_argument(
        "--pixel-patch-size",
        type=int,
        default=_cfg_get(decoder_cfg, "pixel_decoder.patch_size", 16),
    )
    parser.add_argument(
        "--pixel-hidden-dim",
        type=int,
        default=_cfg_get(decoder_cfg, "pixel_decoder.hidden_dim", 512),
    )
    parser.add_argument(
        "--pixel-num-layers",
        type=int,
        default=_cfg_get(decoder_cfg, "pixel_decoder.num_layers", 4),
    )
    parser.add_argument(
        "--pixel-num-heads",
        type=int,
        default=_cfg_get(decoder_cfg, "pixel_decoder.num_heads", 8),
    )
    parser.add_argument(
        "--pixel-mlp-ratio",
        type=float,
        default=_cfg_get(decoder_cfg, "pixel_decoder.mlp_ratio", 4.0),
    )
    parser.add_argument(
        "--pixel-dropout",
        type=float,
        default=_cfg_get(decoder_cfg, "pixel_decoder.dropout", 0.0),
    )
    parser.add_argument(
        "--force-contact-threshold",
        type=float,
        default=_cfg_get(decoder_cfg, "evaluation.force_contact_threshold", 1.0),
        help=(
            "Threshold on decoded force_torque vector norm for drift proxy metrics. "
            "This is in the decoder target scale, usually the model-normalized scale."
        ),
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=_cfg_get(decoder_cfg, "evaluation.contact_threshold", 0.5),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_cfg_get(decoder_cfg, "output_dir"),
    )
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error("checkpoint is required, either as an argument or in decoder config.")
    args.decoder_runtime_config = decoder_cfg
    args.decoder_config_path = decoder_config_path
    return args


def parse_loss_weights(items: list[str]) -> dict[str, float]:
    weights = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected loss weight as KEY=VALUE, got '{item}'.")
        key, value = item.split("=", 1)
        weights[key] = float(value)
    return weights


def default_targets(cfg, available_columns: list[str]) -> list[str]:
    targets = []
    for name, mod_cfg in get_enabled_modality_configs(cfg.obs_encoder).items():
        source = str(mod_cfg.get("source", name))
        if source in available_columns and source not in targets:
            targets.append(source)
    if "bool_contact" in available_columns:
        targets.append("bool_contact")
    return targets


def validate_targets(targets: list[str], available_columns: list[str]) -> None:
    missing = [target for target in targets if target not in available_columns]
    if missing:
        raise ValueError(
            f"Decoder target(s) missing from dataset: {missing}. "
            f"Available columns: {available_columns}"
        )


def build_split_indices(
    dataset_len: int,
    *,
    max_samples: int | None,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}.")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}.")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1.0.")

    sample_count = dataset_len if max_samples is None else min(dataset_len, max_samples)
    if sample_count < 3:
        raise ValueError("Need at least 3 samples to create train/val/test splits.")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_len, generator=generator)[:sample_count]
    train_end = int(sample_count * train_fraction)
    val_end = train_end + int(sample_count * val_fraction)
    if train_end == 0 or val_end <= train_end or val_end >= sample_count:
        raise ValueError(
            f"Invalid split sizes from sample_count={sample_count}, "
            f"train_fraction={train_fraction}, val_fraction={val_fraction}."
        )
    return {
        "train": indices[:train_end].tolist(),
        "val": indices[train_end:val_end].tolist(),
        "test": indices[val_end:].tolist(),
    }


def make_loader(dataset, indices, *, batch_size: int, shuffle: bool, num_workers: int):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def copy_model_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    copied = {}
    for key, value in batch.items():
        copied[key] = value.clone() if torch.is_tensor(value) else value
    if "action" in copied:
        copied["action"] = torch.nan_to_num(copied["action"], 0.0)
    return copied


@torch.no_grad()
def compute_latents(model, batch, cfg) -> dict[str, torch.Tensor | slice]:
    output = model.encode(copy_model_batch(batch))
    emb = output["emb"].detach()
    act_emb = output.get("act_emb")
    if act_emb is None:
        raise KeyError(
            "Model.encode did not return act_emb; action is needed for future prediction."
        )

    history_size = int(cfg.wm.history_size)
    num_preds = int(cfg.wm.num_preds)
    ctx_emb = emb[:, :history_size]
    ctx_act = act_emb[:, :history_size]
    pred_emb = model.predict(ctx_emb, ctx_act).detach()
    future_slice = slice(num_preds, num_preds + pred_emb.size(1))
    encoded_future = emb[:, future_slice].detach()
    return {
        "encoded_all": emb,
        "encoded_future": encoded_future,
        "predicted_future": pred_emb,
        "future_slice": future_slice,
    }


def slice_target_batch(batch, specs, step_slice: slice | None):
    if step_slice is None:
        return {name: batch[name] for name in specs}
    return {name: batch[name][:, step_slice] for name in specs}


def accumulate_metrics(
    accumulator: dict[str, float],
    metrics: dict[str, torch.Tensor],
    *,
    weight: int,
) -> None:
    for key, value in scalar_metrics_to_float(metrics).items():
        accumulator[key] = accumulator.get(key, 0.0) + value * weight


def finalize_metrics(accumulator: dict[str, float], total_weight: int) -> dict[str, float]:
    if total_weight <= 0:
        raise ValueError("No samples were accumulated.")
    return {key: value / total_weight for key, value in accumulator.items()}


def _wandb_enabled(decoder_cfg) -> bool:
    return bool(_cfg_get(decoder_cfg, "wandb.enabled", False))


def _wandb_image_targets(decoder_cfg):
    return _cfg_list(_cfg_get(decoder_cfg, "wandb.image_targets"))


def setup_wandb_logger(args, cfg, targets, *, resolved_model_config: Path):
    decoder_cfg = args.decoder_runtime_config
    if not _wandb_enabled(decoder_cfg):
        return None

    from lightning.pytorch.loggers import WandbLogger

    wandb_config = _cfg_get(decoder_cfg, "wandb.config", {}) or {}
    wandb_config = _plain_config_value(wandb_config)
    wandb_config = {
        key: value for key, value in dict(wandb_config).items() if value is not None
    }
    run_stem = Path(str(args.checkpoint)).name
    wandb_config.setdefault("name", f"decoder_{run_stem}")
    wandb_config.setdefault("resume", "allow")
    wandb_config.setdefault("log_model", False)

    logger = WandbLogger(**wandb_config)
    logger.log_hyperparams(
        {
            "checkpoint": args.checkpoint,
            "decoder_config": str(args.decoder_config_path),
            "resolved_model_config": str(resolved_model_config),
            "dataset_name": cfg.data.dataset.name,
            "targets": targets,
            "train_on": args.train_on,
            "max_samples": args.max_samples,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "num_epochs": args.num_epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
            "pixel_decoder": {
                "image_size": args.pixel_image_size,
                "patch_size": args.pixel_patch_size,
                "hidden_dim": args.pixel_hidden_dim,
                "num_layers": args.pixel_num_layers,
                "num_heads": args.pixel_num_heads,
                "mlp_ratio": args.pixel_mlp_ratio,
                "dropout": args.pixel_dropout,
            },
        }
    )
    return logger


def log_wandb_metrics(logger, prefix: str, metrics: dict[str, float], *, epoch: int):
    if logger is None:
        return
    payload = {f"{prefix}/{key}": value for key, value in metrics.items()}
    payload["epoch"] = epoch
    logger.experiment.log(payload)


def log_wandb_nested_metrics(logger, prefix: str, metrics, *, step: int):
    if logger is None:
        return
    payload = {}
    for split, split_metrics in metrics.items():
        for mode, mode_metrics in split_metrics.items():
            for key, value in mode_metrics.items():
                payload[f"{prefix}/{split}/{mode}/{key}"] = value
    if payload:
        payload["final_step"] = step
        logger.experiment.log(payload)


def _display_channels_first_image(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().float().cpu()
    image = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor shaped (C,H,W), got {image.shape}.")

    channels = image.shape[0]
    if channels == 1:
        return image.repeat(3, 1, 1)
    if channels == 2:
        image = torch.cat([image[0:1], image[1:2]], dim=-1)
        return image.repeat(3, 1, 1)
    return image[:3]


def _truth_prediction_panel(
    truth: torch.Tensor,
    prediction: torch.Tensor,
    spec,
) -> np.ndarray:
    truth = _display_channels_first_image(truth)
    prediction = _display_channels_first_image(prediction)

    if spec.image_normalization == "unit":
        truth = truth.clamp(0.0, 1.0)
        prediction = prediction.clamp(0.0, 1.0)
    else:
        lo = torch.minimum(truth.amin(), prediction.amin())
        hi = torch.maximum(truth.amax(), prediction.amax())
        denom = (hi - lo).clamp_min(1e-6)
        truth = (truth - lo) / denom
        prediction = (prediction - lo) / denom

    gap = torch.ones(3, truth.shape[-2], 4, dtype=truth.dtype) * 0.5
    panel = torch.cat([truth, gap, prediction], dim=-1)
    panel = (panel.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return panel


@torch.no_grad()
def log_wandb_reconstruction_images(
    *,
    logger,
    decoder,
    model,
    loader,
    specs,
    cfg,
    device,
    epoch: int,
    num_images: int,
    image_targets: list[str] | None,
):
    if logger is None or num_images <= 0:
        return

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is required when wandb.enabled=true.") from exc

    selected_targets = image_targets or [
        name for name, spec in specs.items() if spec.kind == "image"
    ]
    selected_targets = [
        name for name in selected_targets if name in specs and specs[name].kind == "image"
    ]
    if not selected_targets:
        return

    decoder.eval()
    model.eval()
    batch = next(iter(loader))
    batch = batch_to_device(batch, device)
    latents = compute_latents(model, batch, cfg)
    target_batch = slice_target_batch(batch, specs, latents["future_slice"])
    predictions = decoder(latents["predicted_future"])

    payload = {}
    for name in selected_targets:
        spec = specs[name]
        prediction = predictions[name]
        target = prepare_decoder_target(target_batch[name], prediction, spec)

        flat_prediction = prediction.reshape(-1, *prediction.shape[-3:])
        flat_target = target.reshape(-1, *target.shape[-3:])
        count = min(int(num_images), flat_prediction.shape[0])
        images = []
        for idx in range(count):
            panel = _truth_prediction_panel(
                flat_target[idx],
                flat_prediction[idx],
                spec,
            )
            images.append(
                wandb.Image(
                    panel,
                    caption=(
                        f"epoch={epoch} target={name} sample={idx} "
                        "left=truth right=predicted_future"
                    ),
                )
            )
        if images:
            payload[f"reconstruction/{name}"] = images

    if payload:
        payload["reconstruction/epoch"] = epoch
        logger.experiment.log(payload)


def run_decoder_epoch(
    *,
    decoder,
    model,
    loader,
    specs,
    cfg,
    device,
    train_on: str,
    optimizer=None,
) -> dict[str, float]:
    is_train = optimizer is not None
    decoder.train(is_train)
    model.eval()

    accumulator = {}
    total_weight = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        latents = compute_latents(model, batch, cfg)
        if train_on == "all":
            z = latents["encoded_all"]
            target_batch = slice_target_batch(batch, specs, None)
        elif train_on == "future":
            z = latents["encoded_future"]
            target_batch = slice_target_batch(batch, specs, latents["future_slice"])
        else:
            raise ValueError(f"Unsupported train_on value: {train_on}")

        z = z.detach()
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        predictions = decoder(z)
        loss, metrics = decoder_loss_and_metrics(predictions, target_batch, specs)

        if is_train:
            loss.backward()
            optimizer.step()

        weight = int(z.shape[0] * z.shape[1])
        accumulate_metrics(accumulator, metrics, weight=weight)
        total_weight += weight

    return finalize_metrics(accumulator, total_weight)


def contact_drift_metrics(
    predictions: dict[str, torch.Tensor],
    target_batch: dict[str, torch.Tensor],
    specs,
    *,
    contact_threshold: float,
    force_contact_threshold: float,
) -> dict[str, torch.Tensor]:
    if "bool_contact" not in predictions or "force_torque" not in predictions:
        return {}

    contact_target = prepare_decoder_target(
        target_batch["bool_contact"],
        predictions["bool_contact"],
        specs["bool_contact"],
    )
    contact_prob = torch.sigmoid(predictions["bool_contact"])
    contact_pred = contact_prob >= contact_threshold

    force = predictions["force_torque"].float()
    force_norm = torch.linalg.norm(force.reshape(*force.shape[:-1], -1), dim=-1)
    force_contact = force_norm >= force_contact_threshold
    contact_pred_flat = contact_pred.squeeze(-1)
    contact_target_flat = contact_target.squeeze(-1).bool()

    disagree = contact_pred_flat != force_contact
    force_contact_head_no_contact = force_contact & (~contact_pred_flat)
    head_contact_force_no_contact = contact_pred_flat & (~force_contact)
    return {
        "drift/contact_head_rate": contact_pred_flat.float().mean().detach(),
        "drift/force_contact_rate": force_contact.float().mean().detach(),
        "drift/disagreement_rate": disagree.float().mean().detach(),
        "drift/force_contact_head_no_contact_rate": (
            force_contact_head_no_contact.float().mean().detach()
        ),
        "drift/head_contact_force_no_contact_rate": (
            head_contact_force_no_contact.float().mean().detach()
        ),
        "drift/contact_head_accuracy": (
            contact_pred_flat == contact_target_flat
        ).float().mean().detach(),
        "drift/force_contact_accuracy": (
            force_contact == contact_target_flat
        ).float().mean().detach(),
    }


@torch.no_grad()
def evaluate_future_decoding(
    *,
    decoder,
    model,
    loader,
    specs,
    cfg,
    device,
    contact_threshold: float,
    force_contact_threshold: float,
) -> dict[str, dict[str, float]]:
    decoder.eval()
    model.eval()

    accumulators = {"encoded_future": {}, "predicted_future": {}}
    total_weight = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        latents = compute_latents(model, batch, cfg)
        target_batch = slice_target_batch(batch, specs, latents["future_slice"])
        weight = int(latents["encoded_future"].shape[0] * latents["encoded_future"].shape[1])
        total_weight += weight

        for mode in ("encoded_future", "predicted_future"):
            predictions = decoder(latents[mode])
            _loss, metrics = decoder_loss_and_metrics(predictions, target_batch, specs)
            metrics.update(
                contact_drift_metrics(
                    predictions,
                    target_batch,
                    specs,
                    contact_threshold=contact_threshold,
                    force_contact_threshold=force_contact_threshold,
                )
            )
            accumulate_metrics(accumulators[mode], metrics, weight=weight)

    return {
        mode: finalize_metrics(accumulator, total_weight)
        for mode, accumulator in accumulators.items()
    }


def print_epoch(epoch, train_metrics, val_metrics):
    print(
        f"epoch={epoch:03d} "
        f"train_loss={train_metrics['loss']:.6f} "
        f"val_loss={val_metrics['loss']:.6f}",
        flush=True,
    )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cache_dir = args.cache_dir.expanduser() if args.cache_dir is not None else None
    cfg, config_path = load_experiment_config(
        checkpoint=args.checkpoint,
        cache_dir=cache_dir,
        config_path=args.config,
        dataset_name=args.dataset_name,
    )
    available_columns = get_dataset_columns(cfg.data.dataset.name, cache_dir)
    targets = args.targets or default_targets(cfg, available_columns)
    validate_targets(targets, available_columns)
    wandb_logger = setup_wandb_logger(
        args,
        cfg,
        targets,
        resolved_model_config=config_path,
    )

    loss_weights = parse_loss_weights(args.loss_weight)
    extra_keys_to_load = [
        target for target in targets if target not in cfg.data.dataset.keys_to_load
    ]
    passthrough_keys = [target for target in targets if target in BINARY_TARGETS]
    dataset = build_dataset(
        cfg,
        cache_dir,
        extra_keys_to_load=extra_keys_to_load,
        passthrough_keys=passthrough_keys,
    )

    split_indices = build_split_indices(
        len(dataset),
        max_samples=args.max_samples,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    loaders = {
        split: make_loader(
            dataset,
            indices,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
        )
        for split, indices in split_indices.items()
    }

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_cost_model(args.checkpoint, cache_dir=cache_dir)
    model = model.to(device).eval()
    model.requires_grad_(False)

    sample_batch = next(iter(loaders["train"]))
    sample_batch = batch_to_device(sample_batch, device)
    sample_latents = compute_latents(model, sample_batch, cfg)
    input_dim = int(sample_latents["encoded_all"].shape[-1])
    pixel_image_size = None if args.pixel_image_size <= 0 else int(args.pixel_image_size)
    specs = infer_target_specs(
        cfg,
        sample_batch,
        targets,
        loss_weights=loss_weights,
        pixel_image_size=pixel_image_size,
    )

    decoder = build_decoder_bank(
        cfg,
        specs,
        input_dim=input_dim,
        pixel_patch_size=args.pixel_patch_size,
        pixel_hidden_dim=args.pixel_hidden_dim,
        pixel_num_layers=args.pixel_num_layers,
        pixel_num_heads=args.pixel_num_heads,
        pixel_mlp_ratio=args.pixel_mlp_ratio,
        pixel_dropout=args.pixel_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print(f"Loaded frozen model from {args.checkpoint}", flush=True)
    print(f"Resolved config: {config_path}", flush=True)
    print(f"Decoder config: {args.decoder_config_path}", flush=True)
    print(f"Decoder targets: {targets}", flush=True)
    print(f"Latent dim: {input_dim}", flush=True)

    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.num_epochs + 1):
        train_metrics = run_decoder_epoch(
            decoder=decoder,
            model=model,
            loader=loaders["train"],
            specs=specs,
            cfg=cfg,
            device=device,
            train_on=args.train_on,
            optimizer=optimizer,
        )
        val_metrics = run_decoder_epoch(
            decoder=decoder,
            model=model,
            loader=loaders["val"],
            specs=specs,
            cfg=cfg,
            device=device,
            train_on=args.train_on,
            optimizer=None,
        )
        print_epoch(epoch, train_metrics, val_metrics)
        log_wandb_metrics(wandb_logger, "train", train_metrics, epoch=epoch)
        log_wandb_metrics(wandb_logger, "val", val_metrics, epoch=epoch)
        image_every = int(
            _cfg_get(args.decoder_runtime_config, "wandb.image_log_every_n_epochs", 1)
        )
        if image_every > 0 and epoch % image_every == 0:
            log_wandb_reconstruction_images(
                logger=wandb_logger,
                decoder=decoder,
                model=model,
                loader=loaders["val"],
                specs=specs,
                cfg=cfg,
                device=device,
                epoch=epoch,
                num_images=int(_cfg_get(args.decoder_runtime_config, "wandb.num_images", 4)),
                image_targets=_wandb_image_targets(args.decoder_runtime_config),
            )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = deepcopy(decoder.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs.", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Decoder training finished without a valid checkpoint.")

    decoder.load_state_dict(best_state)
    split_results = {
        split: evaluate_future_decoding(
            decoder=decoder,
            model=model,
            loader=loader,
            specs=specs,
            cfg=cfg,
            device=device,
            contact_threshold=args.contact_threshold,
            force_contact_threshold=args.force_contact_threshold,
        )
        for split, loader in loaders.items()
    }
    log_wandb_nested_metrics(
        wandb_logger,
        "final",
        split_results,
        step=len(history) + 1,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or Path("experiments/results") / f"decoder_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_payload = {
        "state_dict": decoder.state_dict(),
        "target_specs": {name: spec.to_dict() for name, spec in specs.items()},
        "input_dim": input_dim,
        "checkpoint": args.checkpoint,
        "decoder_config": str(args.decoder_config_path),
        "resolved_config": str(config_path),
        "targets": targets,
        "train_on": args.train_on,
        "pixel_decoder": {
            "image_size": pixel_image_size,
            "patch_size": args.pixel_patch_size,
            "hidden_dim": args.pixel_hidden_dim,
            "num_layers": args.pixel_num_layers,
            "num_heads": args.pixel_num_heads,
            "mlp_ratio": args.pixel_mlp_ratio,
            "dropout": args.pixel_dropout,
        },
    }
    torch.save(checkpoint_payload, output_dir / "decoder.pt")

    metrics_payload = {
        "checkpoint": args.checkpoint,
        "decoder_config": str(args.decoder_config_path),
        "resolved_config": str(config_path),
        "dataset_name": cfg.data.dataset.name,
        "targets": targets,
        "target_specs": {name: spec.to_dict() for name, spec in specs.items()},
        "train_on": args.train_on,
        "best_val_loss": best_val_loss,
        "history": history,
        "splits": {
            split: {
                "num_samples": len(split_indices[split]),
                "metrics": metrics,
            }
            for split, metrics in split_results.items()
        },
    }
    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nSaved decoder checkpoint to {output_dir / 'decoder.pt'}", flush=True)
    print(f"Saved metrics to {output_dir / 'metrics.json'}", flush=True)
    print("\nTest predicted-future metrics:", flush=True)
    for key, value in split_results["test"]["predicted_future"].items():
        print(f"  {key}: {value:.6f}", flush=True)

    if wandb_logger is not None:
        wandb_logger.finalize("success")


if __name__ == "__main__":
    main()
