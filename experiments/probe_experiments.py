import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_utils import (
    batch_to_device,
    build_dataset,
    extract_representation,
    get_dataset_columns,
    load_cost_model,
    load_experiment_config,
)
from probes import (
    ProbeTrainingConfig,
    build_probe,
    evaluate_knn_probe,
    evaluate_probe,
    fit_knn_probe,
    train_probe,
)


@dataclass(frozen=True)
class ProbeExperimentSpec:
    name: str
    task_type: str
    required_keys: tuple[str, ...]
    passthrough_keys: tuple[str, ...]
    output_dim: int
    description: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run frozen-latent probe experiments on a trained LeWM checkpoint."
    )
    parser.add_argument("checkpoint", type=str, help="Checkpoint path or run reference.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["ee_position", "contact_no_contact", "object_distance"],
        choices=["ee_position", "contact_no_contact", "object_distance"],
        help="Probe experiments to run. Missing required dataset keys raise an error.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument(
        "--representation",
        type=str,
        default="fused",
        help="Representation to probe: 'fused' or a modality branch name.",
    )
    parser.add_argument(
        "--probe-type",
        type=str,
        default="linear",
        choices=["linear", "mlp", "knn"],
        help="Probe architecture.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="*",
        default=[256, 256],
        help="Hidden dimensions for nonlinear MLP probes.",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--knn-k", type=int, default=16)
    parser.add_argument(
        "--knn-distance",
        type=str,
        default="euclidean",
        choices=["euclidean", "cosine"],
        help="Distance metric for kNN probes.",
    )
    parser.add_argument("--extract-batch-size", type=int, default=64)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50000,
        help="Maximum number of dataset clips to sample for probing.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--probe-step", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def get_experiment_specs():
    return {
        "ee_position": ProbeExperimentSpec(
            name="ee_position",
            task_type="regression",
            required_keys=("ee_position",),
            passthrough_keys=("ee_position",),
            output_dim=3,
            description="Predict end-effector xyz position from the frozen latent.",
        ),
        "contact_no_contact": ProbeExperimentSpec(
            name="contact_no_contact",
            task_type="binary",
            required_keys=("bool_contact",),
            passthrough_keys=("bool_contact",),
            output_dim=1,
            description="Predict binary contact from the ground-truth bool_contact label.",
        ),
        "object_distance": ProbeExperimentSpec(
            name="object_distance",
            task_type="regression",
            required_keys=("ee_position", "object_1_xyz", "object_2_xyz"),
            passthrough_keys=("ee_position", "object_1_xyz", "object_2_xyz"),
            output_dim=2,
            description="Predict distances from the end effector to object_1 and object_2.",
        ),
    }


def validate_requested_experiments(specs, requested, available_columns):
    missing_by_experiment = {}
    for name in requested:
        spec = specs[name]
        missing = [key for key in spec.required_keys if key not in available_columns]
        if missing:
            missing_by_experiment[name] = missing

    if missing_by_experiment:
        message = ["Missing required dataset keys for requested probe experiments:"]
        for name, missing in missing_by_experiment.items():
            message.append(f"  - {name}: {missing}")
        raise ValueError("\n".join(message))


def select_probe_step(cfg, requested_step):
    total_steps = int(cfg.wm.history_size + cfg.wm.num_preds)
    if requested_step is None:
        requested_step = int(cfg.wm.history_size) - 1

    if requested_step < 0:
        requested_step += total_steps

    if not 0 <= requested_step < total_steps:
        raise ValueError(
            f"probe_step must be in [0, {total_steps - 1}], got {requested_step}."
        )
    return requested_step


def build_split_indices(dataset_len, max_samples, train_fraction, val_fraction, seed):
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1.0")

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


def extract_ee_position(batch, probe_step, args):
    ee_position = batch["ee_position"][:, probe_step].float()
    return ee_position[..., :3]


def extract_contact_no_contact(batch, probe_step, args):
    label = batch["bool_contact"][:, probe_step].float()
    return label.unsqueeze(-1)


def extract_object_distance(batch, probe_step, args):
    ee_position = batch["ee_position"][:, probe_step].float()[..., :3]
    object_1 = batch["object_1_xyz"][:, probe_step].float()[..., :3]
    object_2 = batch["object_2_xyz"][:, probe_step].float()[..., :3]
    dist_1 = torch.linalg.norm(object_1 - ee_position, dim=-1, keepdim=True)
    dist_2 = torch.linalg.norm(object_2 - ee_position, dim=-1, keepdim=True)
    return torch.cat([dist_1, dist_2], dim=-1)


TARGET_EXTRACTORS = {
    "ee_position": extract_ee_position,
    "contact_no_contact": extract_contact_no_contact,
    "object_distance": extract_object_distance,
}


def make_probe_loader(features, targets, batch_size, shuffle):
    dataset = TensorDataset(features.float(), targets.float())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def extract_split_data(
    model,
    dataset,
    indices,
    experiment_names,
    *,
    representation,
    probe_step,
    extract_batch_size,
    num_workers,
    device,
    args,
):
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=extract_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    feature_chunks = []
    target_chunks = {name: [] for name in experiment_names}

    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            features = extract_representation(
                model,
                batch,
                representation=representation,
                probe_step=probe_step,
            ).detach().cpu()
            feature_chunks.append(features)

            for name in experiment_names:
                targets = TARGET_EXTRACTORS[name](batch, probe_step, args)
                if targets.ndim == 1:
                    targets = targets.unsqueeze(-1)
                target_chunks[name].append(targets.detach().cpu())

    if not feature_chunks:
        raise ValueError("No data extracted for probe experiments.")

    features = torch.cat(feature_chunks, dim=0)
    outputs = {}
    for name in experiment_names:
        targets = torch.cat(target_chunks[name], dim=0)
        valid_mask = ~torch.isnan(targets).any(dim=1)
        if valid_mask.sum().item() == 0:
            raise ValueError(f"No valid targets found for experiment '{name}'.")
        outputs[name] = {
            "x": features[valid_mask],
            "y": targets[valid_mask],
        }
    return outputs


def print_summary(results):
    for name, result in results.items():
        print(f"\n=== {name} ===")
        print(f"Probe type: {result['probe_type']}")
        print(f"Task type: {result['task_type']}")
        print(f"Train/Val/Test samples: {result['num_train']}/{result['num_val']}/{result['num_test']}")
        print(f"Best val loss: {result['best_val_loss']:.6f}")
        for split_name, metrics in result["metrics"].items():
            metrics_str = ", ".join(f"{k}={v:.6f}" for k, v in metrics.items())
            print(f"{split_name}: {metrics_str}")


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
    specs = get_experiment_specs()
    validate_requested_experiments(specs, args.experiments, available_columns)

    probe_step = select_probe_step(cfg, args.probe_step)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    required_keys = []
    passthrough_keys = []
    for name in args.experiments:
        spec = specs[name]
        for key in spec.required_keys:
            if key not in required_keys:
                required_keys.append(key)
        for key in spec.passthrough_keys:
            if key not in passthrough_keys:
                passthrough_keys.append(key)

    dataset = build_dataset(
        cfg,
        cache_dir,
        extra_keys_to_load=required_keys,
        passthrough_keys=passthrough_keys,
    )
    split_indices = build_split_indices(
        dataset_len=len(dataset),
        max_samples=args.max_samples,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    model = load_cost_model(args.checkpoint, cache_dir=cache_dir)
    model = model.to(device).eval()
    model.requires_grad_(False)

    extracted = {
        split_name: extract_split_data(
            model,
            dataset,
            indices,
            args.experiments,
            representation=args.representation,
            probe_step=probe_step,
            extract_batch_size=args.extract_batch_size,
            num_workers=args.num_workers,
            device=device,
            args=args,
        )
        for split_name, indices in split_indices.items()
    }

    results = {}
    for name in args.experiments:
        spec = specs[name]
        train_data = extracted["train"][name]
        val_data = extracted["val"][name]
        test_data = extracted["test"][name]

        train_loader = make_probe_loader(
            train_data["x"],
            train_data["y"],
            batch_size=args.probe_batch_size,
            shuffle=True,
        )
        val_loader = make_probe_loader(
            val_data["x"],
            val_data["y"],
            batch_size=args.probe_batch_size,
            shuffle=False,
        )
        test_loader = make_probe_loader(
            test_data["x"],
            test_data["y"],
            batch_size=args.probe_batch_size,
            shuffle=False,
        )

        probe = build_probe(
            args.probe_type,
            input_dim=train_data["x"].shape[-1],
            output_dim=spec.output_dim,
            hidden_dims=tuple(args.hidden_dims),
            dropout=args.dropout,
            k=args.knn_k,
            distance=args.knn_distance,
        )
        if args.probe_type == "knn":
            training = fit_knn_probe(
                probe,
                train_data["x"],
                train_data["y"],
                device=device,
            )
            trained_probe = training["probe"]
            train_metrics = evaluate_knn_probe(
                trained_probe,
                train_data["x"],
                train_data["y"],
                task_type=spec.task_type,
                device=device,
                batch_size=args.probe_batch_size,
            )
            val_metrics = evaluate_knn_probe(
                trained_probe,
                val_data["x"],
                val_data["y"],
                task_type=spec.task_type,
                device=device,
                batch_size=args.probe_batch_size,
            )
            test_metrics = evaluate_knn_probe(
                trained_probe,
                test_data["x"],
                test_data["y"],
                task_type=spec.task_type,
                device=device,
                batch_size=args.probe_batch_size,
            )
            best_val_loss = float(val_metrics["loss"])
        else:
            training = train_probe(
                probe,
                train_loader,
                val_loader,
                config=ProbeTrainingConfig(
                    task_type=spec.task_type,
                    num_epochs=args.num_epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    patience=args.patience,
                ),
                device=device,
            )

            trained_probe = training["probe"]
            train_metrics = evaluate_probe(
                trained_probe,
                train_loader,
                task_type=spec.task_type,
                device=device,
            )
            val_metrics = evaluate_probe(
                trained_probe,
                val_loader,
                task_type=spec.task_type,
                device=device,
            )
            test_metrics = evaluate_probe(
                trained_probe,
                test_loader,
                task_type=spec.task_type,
                device=device,
            )
            best_val_loss = float(training["best_val_loss"])

        results[name] = {
            "description": spec.description,
            "probe_type": args.probe_type,
            "task_type": spec.task_type,
            "representation": args.representation,
            "probe_step": probe_step,
            "num_train": int(train_data["x"].shape[0]),
            "num_val": int(val_data["x"].shape[0]),
            "num_test": int(test_data["x"].shape[0]),
            "best_val_loss": best_val_loss,
            "metrics": {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            },
            "history": training["history"],
        }
        if args.probe_type == "knn":
            results[name]["knn"] = {
                "k": int(args.knn_k),
                "distance": args.knn_distance,
            }

    print_summary(results)

    output_path = args.output
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("experiments/results") / f"probe_experiments_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "checkpoint": args.checkpoint,
        "resolved_config": str(config_path),
        "dataset_name": cfg.data.dataset.name,
        "experiments": args.experiments,
        "representation": args.representation,
        "probe_type": args.probe_type,
        "probe_step": probe_step,
        "device": str(device),
        "results": results,
    }
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
