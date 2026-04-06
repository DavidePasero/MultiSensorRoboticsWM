from collections import OrderedDict

import torch
from torch import nn


class BaseFusion(nn.Module):
    def forward(self, modality_embs):
        raise NotImplementedError


class IdentityFusion(BaseFusion):
    """
    A simple fusion that returns the embedding of a single specified modality without modification.
    This is useful for ablation studies or when only one modality is present.
    """
    def __init__(self, modalities, output_dim=None):
        super().__init__()
        self.modalities = tuple(modalities)
        self.output_dim = output_dim

    def forward(self, modality_embs):
        if len(self.modalities) != 1:
            raise ValueError(
                f"IdentityFusion expects exactly one modality, got {self.modalities}."
            )

        name = self.modalities[0]
        if name not in modality_embs:
            raise KeyError(f"Missing modality '{name}' for identity fusion.")

        fused = modality_embs[name]
        if self.output_dim is not None and fused.size(-1) != self.output_dim:
            raise ValueError(
                f"Identity fusion output dim mismatch: expected {self.output_dim}, "
                f"got {fused.size(-1)}."
            )

        return fused, {}


class ConcatProjectFusion(BaseFusion):
    """
    A simple fusion that concatenates the embeddings of all modalities and projects to a common space."""
    def __init__(self, input_dims, output_dim, hidden_dim=512, dropout=0.0):
        super().__init__()
        self.modalities = tuple(input_dims.keys())
        total_dim = sum(input_dims.values())
        self.net = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, modality_embs):
        missing = [name for name in self.modalities if name not in modality_embs]
        if missing:
            raise KeyError(f"Missing modalities for concat fusion: {missing}.")

        fused = torch.cat([modality_embs[name] for name in self.modalities], dim=-1)
        return self.net(fused), {}


class CrossAttentionFusion(BaseFusion):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, modality_embs):
        raise NotImplementedError(
            "CrossAttentionFusion is intentionally scaffolded but not implemented yet."
        )


class MoEFusion(BaseFusion):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, modality_embs):
        raise NotImplementedError(
            "MoEFusion is intentionally scaffolded but not implemented yet."
        )


def build_fusion(cfg, input_dims):
    fusion_type = cfg.type
    ordered_input_dims = OrderedDict(input_dims)

    if fusion_type == "identity":
        return IdentityFusion(
            modalities=ordered_input_dims.keys(),
            output_dim=cfg.get("output_dim"),
        )

    if fusion_type == "concat_project":
        hidden_dim = cfg.get("hidden_dim", max(sum(ordered_input_dims.values()), cfg.output_dim))
        return ConcatProjectFusion(
            input_dims=ordered_input_dims,
            output_dim=cfg.output_dim,
            hidden_dim=hidden_dim,
            dropout=cfg.get("dropout", 0.0),
        )

    if fusion_type == "cross_attention":
        return CrossAttentionFusion()

    if fusion_type == "moe":
        return MoEFusion()

    raise ValueError(f"Unsupported fusion type: {fusion_type}")
