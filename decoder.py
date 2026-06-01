"""Frozen-latent decoders for LeWM observation reconstruction experiments."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from math import log10
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from multimodal import IMAGE_ENCODER_TYPES, VECTOR_ENCODER_TYPES


IMAGE_CHANNEL_COUNTS = (1, 2, 3, 4)
UNIT_RANGE_IMAGE_TARGETS = {"pixels", "tactile"}
BINARY_TARGETS = {"bool_contact", "success"}


@dataclass(frozen=True)
class DecoderTargetSpec:
    name: str
    kind: str
    shape: tuple[int, ...]
    loss_weight: float = 1.0
    image_normalization: str = "none"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shape"] = list(self.shape)
        return payload


def _as_tuple(value) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(v) for v in value)


def _flatten_latents(z: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    if z.ndim < 2:
        raise ValueError(f"Expected latent tensor with at least 2 dims, got {z.shape}.")
    leading_shape = tuple(z.shape[:-1])
    return z.reshape(-1, z.shape[-1]), leading_shape


def _restore_vector_output(
    x: torch.Tensor,
    leading_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
) -> torch.Tensor:
    if not output_shape:
        return x.reshape(*leading_shape)
    return x.reshape(*leading_shape, *output_shape)


def _canonical_image_shape(shape: tuple[int, ...], *, name: str) -> tuple[int, int, int]:
    if len(shape) == 2:
        height, width = shape
        return 1, int(height), int(width)

    if len(shape) != 3:
        raise ValueError(
            f"Image target '{name}' must have shape (C,H,W), (H,W,C), or (H,W); "
            f"got {shape}."
        )

    if shape[0] in IMAGE_CHANNEL_COUNTS:
        channels, height, width = shape
        return int(channels), int(height), int(width)

    if shape[-1] in IMAGE_CHANNEL_COUNTS:
        height, width, channels = shape
        return int(channels), int(height), int(width)

    raise ValueError(
        f"Unable to infer channel dimension for image target '{name}' with shape {shape}."
    )


def _ensure_channels_first_image(
    target: torch.Tensor,
    *,
    expected_channels: int,
) -> torch.Tensor:
    if target.ndim < 3:
        raise ValueError(f"Expected image target with at least 3 dims, got {target.shape}.")

    if target.shape[-3] == expected_channels:
        return target
    if target.shape[-1] == expected_channels:
        permute_order = list(range(target.ndim))
        channel_axis = permute_order.pop(-1)
        permute_order.insert(target.ndim - 3, channel_axis)
        return target.permute(*permute_order)

    if expected_channels == 1 and target.ndim >= 2:
        return target.unsqueeze(-3)

    raise ValueError(
        f"Could not convert image target shape {tuple(target.shape)} to channels-first "
        f"with {expected_channels} channels."
    )


def _numel(shape: tuple[int, ...]) -> int:
    value = 1
    for dim in shape:
        value *= int(dim)
    return int(value)


class MLPVectorDecoder(nn.Module):
    """Mirror of ``MLPVectorEncoder``: latent -> reversed hidden stack -> vector."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_shape: tuple[int, ...],
        hidden_dims: tuple[int, ...] = (128, 128),
    ):
        super().__init__()
        self.output_shape = _as_tuple(output_shape)
        output_dim = _numel(self.output_shape) if self.output_shape else 1
        dims = [int(input_dim), *reversed(_as_tuple(hidden_dims)), output_dim]

        layers = []
        for idx, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if idx < len(dims) - 2:
                layers.extend([nn.LayerNorm(out_dim), nn.GELU()])
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z, leading_shape = _flatten_latents(z)
        output = self.net(z.float())
        return _restore_vector_output(output, leading_shape, self.output_shape)


class BinaryDecoder(nn.Module):
    """Binary classifier head used for contact/success drift measurements."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 128),
        output_dim: int = 1,
    ):
        super().__init__()
        dims = [int(input_dim), *reversed(_as_tuple(hidden_dims)), int(output_dim)]
        layers = []
        for idx, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if idx < len(dims) - 2:
                layers.extend([nn.LayerNorm(out_dim), nn.GELU()])
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z, leading_shape = _flatten_latents(z)
        logits = self.net(z.float())
        return logits.reshape(*leading_shape, -1)


class CNNImageDecoder(nn.Module):
    """Symmetric transposed-CNN decoder for CNN image modalities."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_shape: tuple[int, ...],
        hidden_dims: tuple[int, ...] = (32, 64, 128),
        head_hidden_dim: int | None = None,
        output_activation: str = "none",
        name: str = "image",
    ):
        super().__init__()
        self.output_shape = _canonical_image_shape(_as_tuple(output_shape), name=name)
        self.out_channels, self.out_height, self.out_width = self.output_shape
        self.output_activation = output_activation

        hidden_dims = _as_tuple(hidden_dims)
        if not hidden_dims:
            raise ValueError("CNNImageDecoder requires at least one hidden dimension.")
        if output_activation not in {"none", "sigmoid"}:
            raise ValueError(
                f"Unsupported output_activation '{output_activation}'. "
                "Expected 'none' or 'sigmoid'."
            )

        num_upsamples = len(hidden_dims)
        self.init_height = max(1, self.out_height // (2**num_upsamples))
        self.init_width = max(1, self.out_width // (2**num_upsamples))
        first_channels = hidden_dims[-1]
        projected_dim = first_channels * self.init_height * self.init_width
        head_hidden_dim = head_hidden_dim or max(int(input_dim), projected_dim)

        self.project = nn.Sequential(
            nn.Linear(int(input_dim), int(head_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(head_hidden_dim), projected_dim),
            nn.GELU(),
        )

        channels = list(reversed(hidden_dims))
        layers = []
        for idx, in_channels in enumerate(channels):
            is_last = idx == len(channels) - 1
            out_channels = self.out_channels if is_last else channels[idx + 1]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            if not is_last:
                layers.append(nn.GELU())
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z, leading_shape = _flatten_latents(z)
        x = self.project(z.float())
        x = x.reshape(-1, self.deconv[0].in_channels, self.init_height, self.init_width)
        x = self.deconv(x)
        if x.shape[-2:] != (self.out_height, self.out_width):
            x = F.interpolate(
                x,
                size=(self.out_height, self.out_width),
                mode="bilinear",
                align_corners=False,
            )
        if self.output_activation == "sigmoid":
            x = torch.sigmoid(x)
        return x.reshape(*leading_shape, self.out_channels, self.out_height, self.out_width)


class CrossAttentionDecoderLayer(nn.Module):
    """One patch-query cross-attention block with a residual MLP."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        memory = self.memory_norm(memory)
        attn_output, _ = self.cross_attention(
            query=self.query_norm(queries),
            key=memory,
            value=memory,
            need_weights=False,
        )
        queries = queries + attn_output
        queries = queries + self.mlp(queries)
        return queries


class PatchQueryPixelDecoder(nn.Module):
    """
    Paper-style diagnostic pixel decoder.

    A global latent token is projected to the decoder dimension and used as the
    key/value memory. One learned query token per output patch cross-attends to
    that global representation, then each query is projected to a pixel patch.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_shape: tuple[int, ...],
        patch_size: int = 16,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        output_activation: str = "sigmoid",
        name: str = "pixels",
    ):
        super().__init__()
        self.output_shape = _canonical_image_shape(_as_tuple(output_shape), name=name)
        self.out_channels, self.out_height, self.out_width = self.output_shape
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.output_activation = output_activation

        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}.")
        if self.out_height % self.patch_size != 0 or self.out_width % self.patch_size != 0:
            raise ValueError(
                f"Pixel decoder output shape {(self.out_height, self.out_width)} must "
                f"be divisible by patch_size={self.patch_size}."
            )
        if output_activation not in {"none", "sigmoid"}:
            raise ValueError(
                f"Unsupported output_activation '{output_activation}'. "
                "Expected 'none' or 'sigmoid'."
            )

        self.grid_height = self.out_height // self.patch_size
        self.grid_width = self.out_width // self.patch_size
        self.num_patches = self.grid_height * self.grid_width
        patch_dim = self.patch_size * self.patch_size * self.out_channels

        self.latent_proj = nn.Linear(int(input_dim), self.hidden_dim)
        self.query_tokens = nn.Parameter(torch.empty(1, self.num_patches, self.hidden_dim))
        self.layers = nn.ModuleList(
            [
                CrossAttentionDecoderLayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.to_patch = nn.Linear(self.hidden_dim, patch_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_tokens, std=0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z, leading_shape = _flatten_latents(z)
        memory = self.latent_proj(z.float()).unsqueeze(1)
        queries = self.query_tokens.expand(z.size(0), -1, -1)

        for layer in self.layers:
            queries = layer(queries, memory)

        patches = self.to_patch(self.output_norm(queries))
        patches = patches.reshape(
            z.size(0),
            self.grid_height,
            self.grid_width,
            self.out_channels,
            self.patch_size,
            self.patch_size,
        )
        image = patches.permute(0, 3, 1, 4, 2, 5).reshape(
            z.size(0),
            self.out_channels,
            self.out_height,
            self.out_width,
        )
        if self.output_activation == "sigmoid":
            image = torch.sigmoid(image)
        return image.reshape(
            *leading_shape,
            self.out_channels,
            self.out_height,
            self.out_width,
        )


class DecoderBank(nn.Module):
    """One decoder per requested target, all fed by the same latent representation."""

    def __init__(self, decoders: OrderedDict[str, nn.Module]):
        super().__init__()
        self.decoders = nn.ModuleDict(decoders)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: decoder(z) for name, decoder in self.decoders.items()}


def _modality_configs_by_source(cfg) -> dict[str, Any]:
    obs_cfg = getattr(cfg, "obs_encoder", None)
    if obs_cfg is None or not hasattr(obs_cfg, "modalities"):
        return {}

    configs = {}
    for name, mod_cfg in obs_cfg.modalities.items():
        if not mod_cfg.get("enabled", True):
            continue
        source = mod_cfg.get("source", name)
        configs[str(source)] = mod_cfg
    return configs


def infer_target_specs(
    cfg,
    sample_batch: dict[str, torch.Tensor],
    targets: list[str],
    *,
    loss_weights: dict[str, float] | None = None,
    pixel_image_size: int | None = 224,
) -> OrderedDict[str, DecoderTargetSpec]:
    modality_configs = _modality_configs_by_source(cfg)
    specs = OrderedDict()
    for name in targets:
        if name not in sample_batch:
            raise KeyError(f"Target '{name}' is missing from the sample batch.")

        value = sample_batch[name]
        if not torch.is_tensor(value):
            raise TypeError(f"Target '{name}' must be a tensor, got {type(value)}.")
        if value.ndim < 2:
            raise ValueError(
                f"Expected target '{name}' to have at least batch/time dims, "
                f"got {tuple(value.shape)}."
            )

        mod_cfg = modality_configs.get(name)
        encoder_type = mod_cfg.encoder_type if mod_cfg is not None else None
        target_shape = tuple(int(dim) for dim in value.shape[2:])

        if name in BINARY_TARGETS:
            kind = "binary"
            target_shape = ()
            image_normalization = "none"
        elif encoder_type in IMAGE_ENCODER_TYPES or name in {"pixels", "depth", "tactile"}:
            kind = "image"
            if not target_shape and value.ndim == 3:
                target_shape = (1, int(value.shape[-2]), int(value.shape[-1]))
            target_shape = _canonical_image_shape(target_shape, name=name)
            if name == "pixels" and pixel_image_size is not None:
                target_shape = (
                    target_shape[0],
                    int(pixel_image_size),
                    int(pixel_image_size),
                )
            image_normalization = "unit" if name in UNIT_RANGE_IMAGE_TARGETS else "none"
        elif encoder_type in VECTOR_ENCODER_TYPES or value.ndim <= 3:
            kind = "vector"
            if not target_shape:
                target_shape = (1,)
            image_normalization = "none"
        else:
            raise ValueError(
                f"Cannot infer decoder kind for target '{name}' with shape "
                f"{tuple(value.shape)}."
            )

        specs[name] = DecoderTargetSpec(
            name=name,
            kind=kind,
            shape=target_shape,
            loss_weight=float((loss_weights or {}).get(name, 1.0)),
            image_normalization=image_normalization,
        )
    return specs


def build_decoder_bank(
    cfg,
    specs: OrderedDict[str, DecoderTargetSpec],
    *,
    input_dim: int,
    default_image_hidden_dims: tuple[int, ...] = (32, 64, 128),
    default_vector_hidden_dims: tuple[int, ...] = (128, 128),
    default_binary_hidden_dims: tuple[int, ...] = (128, 128),
    pixel_patch_size: int = 16,
    pixel_hidden_dim: int = 512,
    pixel_num_layers: int = 4,
    pixel_num_heads: int = 8,
    pixel_mlp_ratio: float = 4.0,
    pixel_dropout: float = 0.0,
) -> DecoderBank:
    modality_configs = _modality_configs_by_source(cfg)
    decoders = OrderedDict()
    for name, spec in specs.items():
        mod_cfg = modality_configs.get(name)

        if spec.kind == "image" and name == "pixels":
            decoders[name] = PatchQueryPixelDecoder(
                input_dim=input_dim,
                output_shape=spec.shape,
                patch_size=pixel_patch_size,
                hidden_dim=pixel_hidden_dim,
                num_layers=pixel_num_layers,
                num_heads=pixel_num_heads,
                mlp_ratio=pixel_mlp_ratio,
                dropout=pixel_dropout,
                output_activation="sigmoid"
                if spec.image_normalization == "unit"
                else "none",
                name=name,
            )
        elif spec.kind == "image":
            hidden_dims = (
                tuple(mod_cfg.get("hidden_dims", default_image_hidden_dims))
                if mod_cfg is not None and mod_cfg.encoder_type == "cnn"
                else default_image_hidden_dims
            )
            output_activation = "sigmoid" if spec.image_normalization == "unit" else "none"
            decoders[name] = CNNImageDecoder(
                input_dim=input_dim,
                output_shape=spec.shape,
                hidden_dims=hidden_dims,
                head_hidden_dim=mod_cfg.get("head_hidden_dim") if mod_cfg is not None else None,
                output_activation=output_activation,
                name=name,
            )
        elif spec.kind == "vector":
            hidden_dims = (
                tuple(mod_cfg.get("hidden_dims", default_vector_hidden_dims))
                if mod_cfg is not None
                else default_vector_hidden_dims
            )
            decoders[name] = MLPVectorDecoder(
                input_dim=input_dim,
                output_shape=spec.shape,
                hidden_dims=hidden_dims,
            )
        elif spec.kind == "binary":
            decoders[name] = BinaryDecoder(
                input_dim=input_dim,
                hidden_dims=default_binary_hidden_dims,
                output_dim=1,
            )
        else:
            raise ValueError(f"Unsupported decoder target kind: {spec.kind}")

    return DecoderBank(decoders)


def prepare_decoder_target(
    target: torch.Tensor,
    prediction: torch.Tensor,
    spec: DecoderTargetSpec,
) -> torch.Tensor:
    target = target.float()

    if spec.kind == "binary":
        if target.ndim == prediction.ndim - 1:
            target = target.unsqueeze(-1)
        return target.reshape_as(prediction)

    if spec.kind == "image":
        channels = prediction.shape[-3]
        target = _ensure_channels_first_image(target, expected_channels=channels)
        if target.shape[-2:] != prediction.shape[-2:]:
            leading_shape = tuple(target.shape[:-3])
            target = target.reshape(-1, channels, *target.shape[-2:])
            target = F.interpolate(
                target,
                size=prediction.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).reshape(*leading_shape, channels, *prediction.shape[-2:])
        if spec.image_normalization == "unit" and target.numel() > 0 and target.max() > 1.5:
            target = target / 255.0
        return target.reshape_as(prediction)

    if spec.kind == "vector":
        if target.ndim == prediction.ndim - 1 and prediction.shape[-1] == 1:
            target = target.unsqueeze(-1)
        return target.reshape_as(prediction)

    raise ValueError(f"Unsupported decoder target kind: {spec.kind}")


def decoder_loss_and_metrics(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    specs: OrderedDict[str, DecoderTargetSpec],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total_loss = None
    metrics = {}

    for name, spec in specs.items():
        prediction = predictions[name]
        target = prepare_decoder_target(batch[name], prediction, spec)

        if spec.kind == "binary":
            loss = F.binary_cross_entropy_with_logits(prediction, target)
            probs = torch.sigmoid(prediction)
            pred_labels = (probs >= 0.5).float()
            accuracy = (pred_labels == target).float().mean()
            positive_rate = target.mean()
            metrics[f"{name}/bce"] = loss.detach()
            metrics[f"{name}/accuracy"] = accuracy.detach()
            metrics[f"{name}/positive_rate"] = positive_rate.detach()
        else:
            mse = F.mse_loss(prediction, target)
            mae = F.l1_loss(prediction, target)
            loss = mse
            metrics[f"{name}/mse"] = mse.detach()
            metrics[f"{name}/mae"] = mae.detach()
            if spec.kind == "image" and spec.image_normalization == "unit":
                psnr = -10.0 * torch.log10(mse.detach().clamp_min(1e-12))
                metrics[f"{name}/psnr"] = psnr

        weighted_loss = spec.loss_weight * loss
        metrics[f"{name}/loss"] = weighted_loss.detach()
        total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss

    if total_loss is None:
        raise ValueError("No decoder predictions were provided.")

    metrics["loss"] = total_loss.detach()
    return total_loss, metrics


def scalar_metrics_to_float(metrics: dict[str, torch.Tensor | float]) -> dict[str, float]:
    return {
        key: float(value.detach().cpu().item() if torch.is_tensor(value) else value)
        for key, value in metrics.items()
    }


def image_psnr_from_mse(mse: float) -> float:
    return -10.0 * log10(max(float(mse), 1e-12))
