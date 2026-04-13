from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPProbe(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers = []
        for idx, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            is_last = idx == len(dims) - 2
            if not is_last:
                layers.extend(
                    [
                        nn.LayerNorm(out_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KNNProbe:
    def __init__(
        self,
        *,
        output_dim: int,
        k: int = 16,
        distance: str = "euclidean",
    ):
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}.")
        if distance not in {"euclidean", "cosine"}:
            raise ValueError(
                f"Unsupported distance metric '{distance}'. "
                "Expected one of: ['euclidean', 'cosine']."
            )

        self.output_dim = output_dim
        self.k = int(k)
        self.distance = distance
        self.train_x = None
        self.train_y = None
        self.device = None

    def fit(self, x: torch.Tensor, y: torch.Tensor, *, device: torch.device):
        x = torch.as_tensor(x).float()
        y = torch.as_tensor(y).float()

        if x.ndim != 2:
            raise ValueError(
                f"KNNProbe expects features with shape (N, D), got {tuple(x.shape)}."
            )
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if y.ndim != 2:
            raise ValueError(
                f"KNNProbe expects targets with shape (N, O), got {tuple(y.shape)}."
            )
        if x.size(0) != y.size(0):
            raise ValueError(
                f"KNNProbe feature/target count mismatch: {x.size(0)} != {y.size(0)}."
            )
        if y.size(1) != self.output_dim:
            raise ValueError(
                f"KNNProbe target dim mismatch: expected {self.output_dim}, got {y.size(1)}."
            )

        self.device = device
        self.train_x = x.to(device, non_blocking=True)
        self.train_y = y.to(device, non_blocking=True)
        return self

    def _neighbor_indices(self, x: torch.Tensor) -> torch.Tensor:
        if self.train_x is None or self.train_y is None:
            raise RuntimeError("KNNProbe must be fit before calling predict.")

        k = min(self.k, self.train_x.size(0))
        if k <= 0:
            raise ValueError("KNNProbe has no training samples to query against.")

        if self.distance == "euclidean":
            distances = torch.cdist(x, self.train_x)
            return torch.topk(distances, k=k, dim=1, largest=False).indices

        x_norm = F.normalize(x, dim=-1)
        bank_norm = F.normalize(self.train_x, dim=-1)
        similarities = x_norm @ bank_norm.T
        return torch.topk(similarities, k=k, dim=1, largest=True).indices

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x).float()
        if x.ndim != 2:
            raise ValueError(
                f"KNNProbe expects query features with shape (N, D), got {tuple(x.shape)}."
            )

        device = self.device or x.device
        x = x.to(device, non_blocking=True)
        neighbor_idx = self._neighbor_indices(x)
        neighbor_targets = self.train_y[neighbor_idx]
        return neighbor_targets.mean(dim=1)


def build_probe(
    probe_type: str,
    input_dim: int,
    output_dim: int,
    *,
    hidden_dims: tuple[int, ...] = (256, 256),
    dropout: float = 0.0,
    k: int = 16,
    distance: str = "euclidean",
):
    if probe_type == "linear":
        return LinearProbe(input_dim=input_dim, output_dim=output_dim)
    if probe_type == "mlp":
        return MLPProbe(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    if probe_type == "knn":
        return KNNProbe(output_dim=output_dim, k=k, distance=distance)
    raise ValueError(f"Unsupported probe_type: {probe_type}")


@dataclass
class ProbeTrainingConfig:
    task_type: str
    num_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10


def _binary_probs_to_logits(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.clamp(1e-6, 1 - 1e-6)
    return torch.logit(probs)


def _as_target_shape(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim == 1:
        targets = targets.unsqueeze(-1)
    if logits.ndim == 1:
        logits = logits.unsqueeze(-1)
    if logits.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch between logits {tuple(logits.shape)} and targets "
            f"{tuple(targets.shape)}."
        )
    return targets


def _loss_fn(task_type: str, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if task_type == "binary":
        targets = _as_target_shape(logits, targets).float()
        return F.binary_cross_entropy_with_logits(logits, targets)
    if task_type == "regression":
        targets = _as_target_shape(logits, targets).float()
        return F.mse_loss(logits, targets)
    raise ValueError(f"Unsupported task_type: {task_type}")


def _compute_metrics(task_type: str, logits: torch.Tensor, targets: torch.Tensor) -> dict:
    if task_type == "binary":
        targets = _as_target_shape(logits, targets).float()
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        tp = ((preds == 1) & (targets == 1)).sum().item()
        fp = ((preds == 1) & (targets == 0)).sum().item()
        fn = ((preds == 0) & (targets == 1)).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (preds == targets).float().mean().item()
        bce = F.binary_cross_entropy_with_logits(logits, targets).item()
        positive_rate = targets.mean().item()
        return {
            "loss": bce,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "positive_rate": positive_rate,
        }

    if task_type == "regression":
        targets = _as_target_shape(logits, targets).float()
        mse = F.mse_loss(logits, targets).item()
        mae = F.l1_loss(logits, targets).item()
        rmse = mse ** 0.5
        return {
            "loss": mse,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
        }

    raise ValueError(f"Unsupported task_type: {task_type}")


def _predictions_to_metric_input(task_type: str, predictions: torch.Tensor) -> torch.Tensor:
    if task_type == "binary":
        return _binary_probs_to_logits(predictions.float())
    if task_type == "regression":
        return predictions.float()
    raise ValueError(f"Unsupported task_type: {task_type}")


def _run_epoch(
    probe: nn.Module,
    loader,
    *,
    task_type: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    is_train = optimizer is not None
    probe.train(is_train)

    all_logits = []
    all_targets = []
    total_loss = 0.0
    total_count = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float()

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = probe(x)
        loss = _loss_fn(task_type, logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    if total_count == 0:
        raise ValueError("Empty dataloader provided to probe training/evaluation.")

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = _compute_metrics(task_type, logits, targets)
    metrics["loss"] = total_loss / total_count
    return metrics


def train_probe(
    probe: nn.Module,
    train_loader,
    val_loader,
    *,
    config: ProbeTrainingConfig,
    device: torch.device,
) -> dict:
    probe = probe.to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_state = None
    best_val_loss = float("inf")
    history = []
    epochs_without_improvement = 0

    for epoch in range(config.num_epochs):
        train_metrics = _run_epoch(
            probe,
            train_loader,
            task_type=config.task_type,
            device=device,
            optimizer=optimizer,
        )
        val_metrics = _run_epoch(
            probe,
            val_loader,
            task_type=config.task_type,
            device=device,
            optimizer=None,
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = deepcopy(probe.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if config.patience > 0 and epochs_without_improvement >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("Probe training finished without a valid best checkpoint.")

    probe.load_state_dict(best_state)
    return {
        "probe": probe,
        "best_val_loss": best_val_loss,
        "history": history,
    }


def fit_knn_probe(
    probe: KNNProbe,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    *,
    device: torch.device,
) -> dict:
    probe.fit(train_features, train_targets, device=device)
    return {
        "probe": probe,
        "best_val_loss": float("nan"),
        "history": [],
    }


@torch.no_grad()
def evaluate_probe(
    probe: nn.Module,
    loader,
    *,
    task_type: str,
    device: torch.device,
) -> dict:
    probe = probe.to(device)
    return _run_epoch(
        probe,
        loader,
        task_type=task_type,
        device=device,
        optimizer=None,
    )


@torch.no_grad()
def evaluate_knn_probe(
    probe: KNNProbe,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    task_type: str,
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    features = torch.as_tensor(features).float()
    targets = torch.as_tensor(targets).float()

    if features.ndim != 2:
        raise ValueError(
            f"KNNProbe expects features with shape (N, D), got {tuple(features.shape)}."
        )
    if targets.ndim == 1:
        targets = targets.unsqueeze(-1)
    if targets.ndim != 2:
        raise ValueError(
            f"KNNProbe expects targets with shape (N, O), got {tuple(targets.shape)}."
        )
    if features.size(0) != targets.size(0):
        raise ValueError(
            f"KNNProbe feature/target count mismatch: {features.size(0)} != {targets.size(0)}."
        )

    if features.size(0) == 0:
        raise ValueError("Empty feature tensor provided to evaluate_knn_probe.")

    pred_chunks = []
    for start in range(0, features.size(0), batch_size):
        x = features[start : start + batch_size].to(device, non_blocking=True)
        preds = probe.predict(x).detach().cpu()
        pred_chunks.append(preds)

    predictions = torch.cat(pred_chunks, dim=0)
    metric_input = _predictions_to_metric_input(task_type, predictions)
    return _compute_metrics(task_type, metric_input, targets.cpu())
