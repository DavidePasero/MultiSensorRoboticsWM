"""Evaluate frozen diagnostic decoders under missing-modality conditions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from decoder import DecoderTargetSpec, build_decoder_bank, decoder_loss_and_metrics
from experiments.experiment_utils import (
    batch_to_device,
    build_dataset,
    get_dataset_columns,
    get_dataset_path,
    load_cost_model,
    load_experiment_config,
    resolve_checkpoint_path,
)
from train_decoder import (
    BINARY_TARGETS,
    accumulate_metrics,
    contact_drift_metrics,
    copy_model_batch,
    finalize_metrics,
    parse_loss_weights,
    slice_target_batch,
    validate_targets,
)
from multimodal import get_enabled_modality_configs


ERROR_METRIC_SUFFIXES = ("/mse", "/mae", "/loss", "/bce")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved decoder.pt checkpoints on a full dataset while masking "
            "modalities before the frozen world model encodes observations."
        )
    )
    parser.add_argument(
        "--decoder-dir",
        type=Path,
        default=Path.home() / ".stable_world_model" / "decoders",
        help="Directory containing decoder.pt files, searched recursively.",
    )
    parser.add_argument(
        "--decoder",
        action="append",
        type=Path,
        default=[],
        help="Explicit decoder.pt path. Can be passed multiple times.",
    )
    parser.add_argument("--decoder-glob", default="**/decoder.pt")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", type=str, default="metaworld_eval_button_press")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--loss-weight", action="append", default=[])
    parser.add_argument("--keep-modalities", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--contact-threshold", type=float, default=0.5)
    parser.add_argument("--force-contact-threshold", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--fail-on-skipped",
        action="store_true",
        help="Raise an error if a condition cannot run instead of recording it as skipped.",
    )
    return parser.parse_args()


def decoder_paths(args) -> list[Path]:
    paths = [path.expanduser() for path in args.decoder]
    decoder_dir = args.decoder_dir.expanduser()
    if decoder_dir.exists():
        paths.extend(sorted(decoder_dir.glob(args.decoder_glob)))

    unique = []
    seen = set()
    for path in paths:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)

    if not unique:
        raise FileNotFoundError(
            f"No decoder checkpoints found. Looked in {decoder_dir} "
            f"with glob {args.decoder_glob!r}."
        )
    return unique


def decoder_label(path: Path) -> str:
    if path.name == "decoder.pt":
        return path.parent.name
    return path.stem


def resolve_checkpoint_reference(checkpoint: str, cache_dir: Path | None) -> str:
    try:
        return str(resolve_checkpoint_path(checkpoint, cache_dir))
    except (FileNotFoundError, AssertionError):
        pass

    root = cache_dir.expanduser() if cache_dir is not None else Path.home() / ".stable_worldmodel"
    stem = Path(checkpoint).name
    filename = stem if stem.endswith("_object.ckpt") else f"{stem}_object.ckpt"
    candidates = sorted(root.glob(f"**/{filename}"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not resolve checkpoint {checkpoint!r} under {root}."
        )

    requested_parts = set(Path(checkpoint).parts)
    preferred = [
        candidate for candidate in candidates if candidate.parent.name in requested_parts
    ]
    return str((preferred or candidates)[-1])


def spec_from_payload(payload: dict) -> OrderedDict[str, DecoderTargetSpec]:
    specs = OrderedDict()
    for name, raw_spec in payload["target_specs"].items():
        specs[name] = DecoderTargetSpec(
            name=raw_spec.get("name", name),
            kind=raw_spec["kind"],
            shape=tuple(raw_spec["shape"]),
            loss_weight=float(raw_spec.get("loss_weight", 1.0)),
            image_normalization=raw_spec.get("image_normalization", "none"),
        )
    return specs


def load_decoder(path: Path, cfg, device: torch.device):
    payload = torch.load(path, weights_only=False, map_location="cpu")
    specs = spec_from_payload(payload)
    pixel_cfg = payload.get("pixel_decoder", {}) or {}
    decoder = build_decoder_bank(
        cfg,
        specs,
        input_dim=int(payload["input_dim"]),
        pixel_patch_size=int(pixel_cfg.get("patch_size", 16)),
        pixel_hidden_dim=int(pixel_cfg.get("hidden_dim", 512)),
        pixel_num_layers=int(pixel_cfg.get("num_layers", 4)),
        pixel_num_heads=int(pixel_cfg.get("num_heads", 8)),
        pixel_mlp_ratio=float(pixel_cfg.get("mlp_ratio", 4.0)),
        pixel_dropout=float(pixel_cfg.get("dropout", 0.0)),
    )
    decoder.load_state_dict(payload["state_dict"])
    return decoder.to(device).eval(), specs, payload


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
    if encoders is not None:
        sources = []
        for modality_encoder in encoders.values():
            source = str(getattr(modality_encoder, "source", ""))
            if source and source not in sources:
                sources.append(source)
        return sources
    return []


def config_modality_sources(cfg) -> list[str]:
    sources = []
    for name, mod_cfg in get_enabled_modality_configs(cfg.obs_encoder).items():
        source = str(mod_cfg.get("source", name))
        if source not in sources:
            sources.append(source)
    return sources


def build_conditions(sources: list[str], keep_modalities: list[str] | None):
    conditions = [("all_modalities", [])]
    for source in sources:
        if len(sources) > 1:
            conditions.append((f"drop_{source}", [source]))

    if keep_modalities is not None:
        unknown = [modality for modality in keep_modalities if modality not in sources]
        if unknown:
            raise ValueError(
                f"Unknown keep modalities: {unknown}. Available: {sources}."
            )
        drop = [source for source in sources if source not in keep_modalities]
        label = "keep_" + "_".join(keep_modalities) if keep_modalities else "keep_none"
        if drop and len(drop) < len(sources):
            condition = (label, drop)
            if condition not in conditions:
                conditions.append(condition)
    return conditions


def compute_masked_latents(model, batch, cfg, drop_modalities: list[str]):
    model_batch = copy_model_batch(batch)
    for modality in drop_modalities:
        model_batch.pop(modality, None)

    output = model.encode(model_batch)
    emb = output["emb"].detach()
    act_emb = output.get("act_emb")
    if act_emb is None:
        raise KeyError("Model.encode did not return act_emb.")

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


@torch.no_grad()
def evaluate_condition(
    *,
    decoder,
    model,
    loader,
    specs,
    cfg,
    device,
    drop_modalities: list[str],
    contact_threshold: float,
    force_contact_threshold: float,
):
    accumulators = {
        "encoded_all": {},
        "encoded_future": {},
        "predicted_future": {},
    }
    weights = {mode: 0 for mode in accumulators}

    decoder.eval()
    model.eval()

    for batch in loader:
        batch = batch_to_device(batch, device)
        latents = compute_masked_latents(model, batch, cfg, drop_modalities)

        mode_targets = {
            "encoded_all": slice_target_batch(batch, specs, None),
            "encoded_future": slice_target_batch(batch, specs, latents["future_slice"]),
            "predicted_future": slice_target_batch(batch, specs, latents["future_slice"]),
        }

        for mode, target_batch in mode_targets.items():
            predictions = decoder(latents[mode])
            _loss, metrics = decoder_loss_and_metrics(predictions, target_batch, specs)
            if mode == "predicted_future":
                metrics.update(
                    contact_drift_metrics(
                        predictions,
                        target_batch,
                        specs,
                        contact_threshold=contact_threshold,
                        force_contact_threshold=force_contact_threshold,
                    )
                )
            weight = int(latents[mode].shape[0] * latents[mode].shape[1])
            accumulate_metrics(accumulators[mode], metrics, weight=weight)
            weights[mode] += weight

    return {
        mode: finalize_metrics(accumulator, weights[mode])
        for mode, accumulator in accumulators.items()
    }


def build_loader(cfg, cache_dir, specs, sources, args):
    available_columns = get_dataset_columns(cfg.data.dataset.name, cache_dir)
    targets = list(specs)
    validate_targets(targets, available_columns)
    extra_keys = sorted(set(targets) | set(sources) | {"action"})
    passthrough_keys = [target for target in targets if target in BINARY_TARGETS]
    dataset = build_dataset(
        cfg,
        cache_dir,
        extra_keys_to_load=extra_keys,
        passthrough_keys=passthrough_keys,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return dataset, loader


def dataset_episode_count(dataset_name: str, cache_dir: Path | None) -> int | None:
    path = get_dataset_path(dataset_name, cache_dir)
    if not path.exists():
        return None
    with h5py.File(path, "r") as f:
        key = "episode_idx" if "episode_idx" in f else "ep_idx" if "ep_idx" in f else None
        if key is None:
            return None
        return int(len(set(f[key][()])))


def flatten_results(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        for condition, condition_metrics in result["conditions"].items():
            if condition_metrics.get("status") != "ok":
                rows.append(
                    {
                        "decoder": result["decoder"],
                        "condition": condition,
                        "mode": "",
                        "metric": "status",
                        "value": condition_metrics.get("status"),
                    }
                )
                continue
            for mode, metrics in condition_metrics["metrics"].items():
                for metric, value in metrics.items():
                    rows.append(
                        {
                            "decoder": result["decoder"],
                            "condition": condition,
                            "mode": mode,
                            "metric": metric,
                            "value": value,
                        }
                    )
    return rows


def degradation_rows(results: list[dict]) -> list[dict]:
    rows = []
    for result in results:
        baseline = result["conditions"].get("all_modalities", {})
        if baseline.get("status") != "ok":
            continue
        baseline_metrics = baseline["metrics"]
        for condition, condition_result in result["conditions"].items():
            if condition == "all_modalities" or condition_result.get("status") != "ok":
                continue
            for mode, metrics in condition_result["metrics"].items():
                base_mode = baseline_metrics.get(mode, {})
                for metric, value in metrics.items():
                    if not (metric == "loss" or metric.endswith(ERROR_METRIC_SUFFIXES)):
                        continue
                    base_value = base_mode.get(metric)
                    if base_value is None:
                        continue
                    rows.append(
                        {
                            "decoder": result["decoder"],
                            "condition": condition,
                            "mode": mode,
                            "metric": metric,
                            "baseline": base_value,
                            "masked": value,
                            "delta": value - base_value,
                            "ratio": value / base_value if base_value != 0 else None,
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    cache_dir = args.cache_dir.expanduser() if args.cache_dir is not None else None
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loss_weights = parse_loss_weights(args.loss_weight)

    output_dir = args.output_dir or (
        REPO_ROOT
        / "documentation"
        / f"decoder_masked_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for decoder_path in decoder_paths(args):
        label = decoder_label(decoder_path)
        payload = torch.load(decoder_path, weights_only=False, map_location="cpu")
        checkpoint = args.checkpoint or payload.get("checkpoint")
        if checkpoint is None:
            raise ValueError(
                f"Decoder payload {decoder_path} does not contain a checkpoint. "
                "Pass --checkpoint explicitly."
            )
        checkpoint = resolve_checkpoint_reference(str(checkpoint), cache_dir)

        payload_config = payload.get("resolved_config")
        config_path = args.config
        if config_path is None and payload_config is not None:
            candidate = Path(str(payload_config)).expanduser()
            config_path = candidate if candidate.exists() else None

        cfg, resolved_config = load_experiment_config(
            checkpoint=checkpoint,
            cache_dir=cache_dir,
            config_path=config_path,
            dataset_name=args.dataset_name,
        )

        model = load_cost_model(checkpoint, cache_dir=cache_dir)
        model = model.to(device).eval()
        model.requires_grad_(False)

        decoder, specs, payload = load_decoder(decoder_path, cfg, device)
        if args.targets is not None:
            missing = [target for target in args.targets if target not in specs]
            if missing:
                raise ValueError(
                    f"Decoder {decoder_path} does not contain target heads {missing}. "
                    f"Available targets: {list(specs)}."
                )
            specs = OrderedDict((target, specs[target]) for target in args.targets)
        if loss_weights:
            specs = OrderedDict(
                (
                    name,
                    DecoderTargetSpec(
                        name=spec.name,
                        kind=spec.kind,
                        shape=spec.shape,
                        loss_weight=float(loss_weights.get(name, spec.loss_weight)),
                        image_normalization=spec.image_normalization,
                    ),
                )
                for name, spec in specs.items()
            )

        sources = model_modality_sources(model) or config_modality_sources(cfg)
        if not sources:
            raise ValueError(f"Could not infer modality sources for {checkpoint}.")

        dataset, loader = build_loader(cfg, cache_dir, specs, sources, args)
        supports_missing = model_supports_missing_modalities(model)
        conditions = build_conditions(sources, args.keep_modalities)

        decoder_result = {
            "decoder": label,
            "decoder_path": str(decoder_path),
            "checkpoint": checkpoint,
            "resolved_config": str(resolved_config),
            "dataset_name": args.dataset_name,
            "num_samples": len(dataset),
            "num_episodes": dataset_episode_count(args.dataset_name, cache_dir),
            "modalities": sources,
            "supports_missing_modalities": supports_missing,
            "targets": list(specs),
            "conditions": OrderedDict(),
        }

        print(f"\nDecoder: {label}", flush=True)
        print(f"  checkpoint: {checkpoint}", flush=True)
        print(f"  modalities: {sources}", flush=True)
        print(f"  targets: {list(specs)}", flush=True)

        for condition_name, drop_modalities in conditions:
            if drop_modalities and not supports_missing:
                message = (
                    "model does not support missing modalities, so masked "
                    "conditions cannot use the imputer"
                )
                if args.fail_on_skipped:
                    raise RuntimeError(f"{label}/{condition_name}: {message}")
                decoder_result["conditions"][condition_name] = {
                    "status": "skipped",
                    "drop_modalities": drop_modalities,
                    "reason": message,
                }
                print(f"  {condition_name}: skipped ({message})", flush=True)
                continue

            if len(drop_modalities) >= len(sources):
                message = "condition drops every available modality"
                if args.fail_on_skipped:
                    raise RuntimeError(f"{label}/{condition_name}: {message}")
                decoder_result["conditions"][condition_name] = {
                    "status": "skipped",
                    "drop_modalities": drop_modalities,
                    "reason": message,
                }
                print(f"  {condition_name}: skipped ({message})", flush=True)
                continue

            print(f"  {condition_name}: drop={drop_modalities}", flush=True)
            metrics = evaluate_condition(
                decoder=decoder,
                model=model,
                loader=loader,
                specs=specs,
                cfg=cfg,
                device=device,
                drop_modalities=drop_modalities,
                contact_threshold=args.contact_threshold,
                force_contact_threshold=args.force_contact_threshold,
            )
            decoder_result["conditions"][condition_name] = {
                "status": "ok",
                "drop_modalities": drop_modalities,
                "metrics": metrics,
            }

            pred_loss = metrics["predicted_future"].get("loss")
            enc_loss = metrics["encoded_all"].get("loss")
            print(
                f"    encoded_all/loss={enc_loss:.6f} "
                f"predicted_future/loss={pred_loss:.6f}",
                flush=True,
            )

        results.append(decoder_result)

    summary = {
        "decoder_dir": str(args.decoder_dir.expanduser()),
        "dataset_name": args.dataset_name,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "keep_modalities": args.keep_modalities,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "results": results,
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    write_csv(output_dir / "metrics.csv", flatten_results(results))
    write_csv(output_dir / "degradation.csv", degradation_rows(results))
    print(f"\nSaved decoder masked eval to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
