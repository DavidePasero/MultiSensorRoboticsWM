"""Fusion modules that operate on completed modality tokens."""

from collections import OrderedDict

import torch
from torch import nn


class BaseFusion(nn.Module):
    supports_missing_modalities = False

    def forward(self, modality_tokens):
        raise NotImplementedError


def _validate_token_dims(input_dims):
    ordered_input_dims = OrderedDict(input_dims)
    if not ordered_input_dims:
        raise ValueError("Fusion requires at least one modality input.")

    dims = list(ordered_input_dims.values())
    token_dim = dims[0]
    if any(dim != token_dim for dim in dims):
        raise ValueError(
            "Fusion expects all modality tokens to share one common dimension. "
            f"Got {ordered_input_dims}."
        )

    return ordered_input_dims, token_dim


class IdentityFusion(BaseFusion):
    """Return the single modality token, optionally projecting to output_dim."""

    def __init__(self, modalities, input_dim, output_dim=None):
        super().__init__()
        self.modalities = tuple(modalities)
        self.input_dim = int(input_dim)
        self.output_dim = None if output_dim is None else int(output_dim)
        self.output_proj = (
            nn.Identity()
            if self.output_dim is None or self.output_dim == self.input_dim
            else nn.Linear(self.input_dim, self.output_dim)
        )

    def forward(self, modality_tokens):
        if len(self.modalities) != 1:
            raise ValueError(
                f"IdentityFusion expects exactly one modality, got {self.modalities}."
            )

        name = self.modalities[0]
        if name not in modality_tokens:
            raise KeyError(f"Missing modality '{name}' for identity fusion.")

        return self.output_proj(modality_tokens[name]), {}


class ConcatProjectFusion(BaseFusion):
    """Concatenate completed modality tokens and project to the world-model latent."""

    def __init__(self, input_dims, output_dim, hidden_dim=512, dropout=0.0):
        super().__init__()
        ordered_input_dims, token_dim = _validate_token_dims(input_dims)
        self.modalities = tuple(ordered_input_dims.keys())
        self.token_dim = token_dim
        total_dim = len(self.modalities) * self.token_dim

        self.net = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, modality_tokens):
        fused = torch.cat([modality_tokens[name] for name in self.modalities], dim=-1)
        return self.net(fused), {}


class FeaturewiseGatedFusion(BaseFusion):
    """
    GMU-inspired gated fusion on completed modality tokens.

    Each modality token receives a feature-wise gate predicted from the token
    itself and the mean token context. A softmax across modalities is applied
    independently per feature dimension.
    """

    def __init__(self, input_dims, output_dim, gate_hidden_dim=256, dropout=0.0):
        super().__init__()
        ordered_input_dims, token_dim = _validate_token_dims(input_dims)
        if gate_hidden_dim <= 0:
            raise ValueError(
                f"gate_hidden_dim must be positive, got {gate_hidden_dim}."
            )

        self.modalities = tuple(ordered_input_dims.keys())
        self.token_dim = token_dim
        self.gate_networks = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(2 * token_dim, gate_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(gate_hidden_dim, token_dim),
                )
                for name in self.modalities
            }
        )
        self.output_norm = nn.LayerNorm(token_dim)
        self.output_proj = (
            nn.Identity()
            if output_dim == token_dim
            else nn.Linear(token_dim, output_dim)
        )

    def forward(self, modality_tokens):
        tokens = torch.stack([modality_tokens[name] for name in self.modalities], dim=2)
        context = tokens.mean(dim=2, keepdim=True).expand_as(tokens)
        gate_inputs = torch.cat([tokens, context], dim=-1)

        gate_logits = []
        for idx, name in enumerate(self.modalities):
            gate_logits.append(self.gate_networks[name](gate_inputs[:, :, idx]))
        gate_logits = torch.stack(gate_logits, dim=2)

        gate_weights = gate_logits.softmax(dim=2)
        fused = (gate_weights * tokens).sum(dim=2)
        fused = self.output_norm(fused)
        fused = self.output_proj(fused)
        return fused, {}


class StateTokenAttentionFusion(BaseFusion):
    """
    Treat each completed modality token as a sequence element and use a learned
    state token to produce the fused latent.
    """

    def __init__(
        self,
        input_dims,
        output_dim,
        num_heads=4,
        num_layers=2,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        ordered_input_dims, token_dim = _validate_token_dims(input_dims)
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim ({token_dim}) must be divisible by num_heads ({num_heads})."
            )
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")

        self.modalities = tuple(ordered_input_dims.keys())
        self.token_dim = token_dim
        self.modality_embeddings = nn.Parameter(
            torch.empty(len(self.modalities), token_dim)
        )
        self.state_token = nn.Parameter(torch.empty(1, 1, token_dim))

        ff_dim = int(token_dim * mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(token_dim)
        self.output_proj = (
            nn.Identity()
            if output_dim == token_dim
            else nn.Linear(token_dim, output_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.modality_embeddings, std=0.02)
        nn.init.normal_(self.state_token, std=0.02)

    def forward(self, modality_tokens):
        tokens = torch.stack([modality_tokens[name] for name in self.modalities], dim=2)
        tokens = tokens + self.modality_embeddings.to(
            device=tokens.device, dtype=tokens.dtype
        ).view(1, 1, len(self.modalities), self.token_dim)

        batch_size, num_steps, num_modalities, token_dim = tokens.shape
        tokens = tokens.reshape(batch_size * num_steps, num_modalities, token_dim)

        state_token = self.state_token.to(
            device=tokens.device, dtype=tokens.dtype
        ).expand(batch_size * num_steps, -1, -1)
        sequence = torch.cat([state_token, tokens], dim=1)
        sequence = self.encoder(sequence)

        fused = self.output_norm(sequence[:, 0])
        fused = self.output_proj(fused)
        fused = fused.reshape(batch_size, num_steps, -1)
        return fused, {}


def build_fusion(cfg, input_dims):
    fusion_type = str(cfg.type)
    ordered_input_dims = OrderedDict(input_dims)

    if fusion_type == "identity":
        _, input_dim = _validate_token_dims(ordered_input_dims)
        return IdentityFusion(
            modalities=ordered_input_dims.keys(),
            input_dim=input_dim,
            output_dim=cfg.get("output_dim"),
        )

    if fusion_type == "concatproject":
        concat_cfg = cfg.get("concatproject", cfg)
        _, token_dim = _validate_token_dims(ordered_input_dims)
        hidden_dim = concat_cfg.get(
            "hidden_dim",
            max(
                sum(ordered_input_dims.values()),
                cfg.output_dim,
                len(ordered_input_dims) * token_dim,
            ),
        )
        return ConcatProjectFusion(
            input_dims=ordered_input_dims,
            output_dim=cfg.output_dim,
            hidden_dim=hidden_dim,
            dropout=concat_cfg.get("dropout", 0.0),
        )

    if fusion_type == "gated":
        gated_cfg = cfg.get("gated", cfg)
        _, token_dim = _validate_token_dims(ordered_input_dims)
        return FeaturewiseGatedFusion(
            input_dims=ordered_input_dims,
            output_dim=cfg.output_dim,
            gate_hidden_dim=gated_cfg.get("gate_hidden_dim", 2 * token_dim),
            dropout=gated_cfg.get("dropout", 0.0),
        )

    if fusion_type == "selfattention":
        selfattention_cfg = cfg.get("selfattention", cfg)
        return StateTokenAttentionFusion(
            input_dims=ordered_input_dims,
            output_dim=cfg.output_dim,
            num_heads=selfattention_cfg.get("num_heads", 4),
            num_layers=selfattention_cfg.get("num_layers", 2),
            mlp_ratio=selfattention_cfg.get("mlp_ratio", 4.0),
            dropout=selfattention_cfg.get("dropout", 0.0),
        )

    raise ValueError(f"Unsupported fusion type: {fusion_type}")
