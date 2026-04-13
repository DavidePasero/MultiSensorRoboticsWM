import argparse
import json
import sys
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_utils import (
    build_dataset,
    get_modalities,
    load_cost_model,
    load_experiment_config,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate modality usage with offline ablations by measuring "
            "latent drift ||z_clean - z_corrupt|| and prediction loss increase."
        )
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help=(
            "Checkpoint reference understood by stable_worldmodel, e.g. "
            "'my_run/lewm_epoch_3' or an absolute path to '..._object.ckpt'."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to the training config.yaml. Inferred when omitted.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional STABLEWM_HOME override for dataset/checkpoint lookup.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional dataset name override. Defaults to the training config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for offline evaluation.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=32,
        help="Number of dataloader batches to evaluate.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader workers. Defaults to 0 for robustness.",
    )
    parser.add_argument(
        "--modalities",
        nargs="*",
        default=None,
        help="Subset of modalities to ablate. Defaults to all enabled modalities.",
    )
    parser.add_argument(
        "--corruptions",
        nargs="+",
        default=["zero", "mean", "shuffle", "gaussian"],
        choices=["zero", "mean", "shuffle", "gaussian"],
        help="Corruption operators to apply for each modality.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for dataloader order and stochastic corruptions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device. Defaults to cuda when available, else cpu.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to experiments/results/<timestamp>.json",
    )
    return parser.parse_args()


def copy_batch(batch):
    return {
        k: v.clone() if torch.is_tensor(v) else deepcopy(v)
        for k, v in batch.items()
    }


def batch_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


def reduce_dims_except_feature(x: torch.Tensor) -> tuple[int, ...]:
    if x.ndim < 3:
        return tuple(range(x.ndim))
    return tuple(i for i in range(x.ndim) if i != 2)


def corrupt_tensor(x: torch.Tensor, corruption: str) -> torch.Tensor:
    if corruption == "zero":
        return torch.zeros_like(x)

    reduce_dims = reduce_dims_except_feature(x)
    if corruption == "mean":
        mean = x.mean(dim=reduce_dims, keepdim=True)
        return mean.expand_as(x)

    if corruption == "gaussian":
        mean = x.mean(dim=reduce_dims, keepdim=True)
        std = x.std(dim=reduce_dims, keepdim=True).clamp_min(1e-6)
        noise = torch.randn_like(x)
        return noise * std + mean

    if corruption == "shuffle":
        if x.size(0) > 1:
            perm = torch.randperm(x.size(0), device=x.device)
            return x[perm]
        if x.ndim > 1 and x.size(1) > 1:
            perm = torch.randperm(x.size(1), device=x.device)
            return x[:, perm]
        return x.flip(0)

    raise ValueError(f"Unsupported corruption: {corruption}")


def apply_corruption(batch, modality: str, corruption: str):
    corrupted = copy_batch(batch)
    corrupted[modality] = corrupt_tensor(corrupted[modality], corruption)
    return corrupted


def compute_pred_metrics(model, batch):
    batch = copy_batch(batch)
    if "action" in batch:
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    outputs = model.encode(batch)
    emb = outputs["emb"]
    act_emb = outputs["act_emb"]

    ctx_len = model.predictor.pos_embedding.size(1)
    n_preds = emb.size(1) - ctx_len
    if n_preds <= 0:
        raise ValueError(
            f"Expected sequence length > ctx_len, got emb.shape={tuple(emb.shape)} "
            f"and ctx_len={ctx_len}."
        )

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = model.predict(ctx_emb, ctx_act)
    pred_loss = (pred_emb - tgt_emb).pow(2).mean()

    return {
        "emb": emb,
        "pred_emb": pred_emb,
        "tgt_emb": tgt_emb,
        "pred_loss": pred_loss,
    }


def summarize_metric(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else float("nan"),
        "std": float(arr.std(ddof=0)) if len(arr) else float("nan"),
        "count": int(len(arr)),
    }


def print_results(results):
    for corruption, corruption_results in results.items():
        print(f"\n=== Corruption: {corruption} ===")
        header = (
            f"{'modality':<12} {'z_l2_mean':>12} {'z_l2_std':>12} "
            f"{'pred_clean':>12} {'pred_corrupt':>14} {'pred_increase':>14}"
        )
        print(header)
        print("-" * len(header))
        for modality, metrics in corruption_results.items():
            print(
                f"{modality:<12} "
                f"{metrics['z_l2']['mean']:>12.6f} "
                f"{metrics['z_l2']['std']:>12.6f} "
                f"{metrics['pred_loss_clean']['mean']:>12.6f} "
                f"{metrics['pred_loss_corrupt']['mean']:>14.6f} "
                f"{metrics['pred_loss_increase']['mean']:>14.6f}"
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

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    model = load_cost_model(args.checkpoint, cache_dir=cache_dir)
    model = model.to(device).eval()
    model.requires_grad_(False)

    dataset = build_dataset(cfg, cache_dir)
    modalities = get_modalities(cfg, dataset, args.modalities)
    if not modalities:
        raise ValueError("No modalities available for ablation.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.num_batches:
                break

            batch = batch_to_device(batch, device)
            clean = compute_pred_metrics(model, batch)

            for corruption in args.corruptions:
                for modality in modalities:
                    if modality not in batch:
                        continue

                    corrupted_batch = apply_corruption(batch, modality, corruption)
                    corrupted = compute_pred_metrics(model, corrupted_batch)

                    z_l2 = (clean["emb"] - corrupted["emb"]).pow(2).sum(dim=-1).sqrt().mean()
                    pred_loss_clean = clean["pred_loss"]
                    pred_loss_corrupt = corrupted["pred_loss"]
                    pred_loss_increase = pred_loss_corrupt - pred_loss_clean

                    metrics[corruption][modality]["z_l2"].append(z_l2.item())
                    metrics[corruption][modality]["pred_loss_clean"].append(pred_loss_clean.item())
                    metrics[corruption][modality]["pred_loss_corrupt"].append(pred_loss_corrupt.item())
                    metrics[corruption][modality]["pred_loss_increase"].append(pred_loss_increase.item())

    results = {}
    for corruption, corruption_metrics in metrics.items():
        results[corruption] = {}
        for modality, modality_metrics in corruption_metrics.items():
            results[corruption][modality] = {
                metric_name: summarize_metric(values)
                for metric_name, values in modality_metrics.items()
            }

    print_results(results)

    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("experiments/results") / f"modality_ablation_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint": args.checkpoint,
        "resolved_config": str(config_path),
        "dataset_name": cfg.data.dataset.name,
        "modalities": modalities,
        "corruptions": args.corruptions,
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "device": str(device),
        "results": results,
    }
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
