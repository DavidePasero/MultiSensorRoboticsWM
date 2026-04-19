from collections import OrderedDict

import torch
from torch import nn


class BaseFusion(nn.Module):
    supports_missing_modalities = False

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


class StateTokenAttentionFusion(BaseFusion):
    """
    Fusion that treats each modality embedding as a token, prepends a learned
    state token, and uses self-attention to produce the fused latent state.
    """

    def __init__(
        self,
        input_dims,
        output_dim,
        model_dim=128,
        num_heads=4,
        num_layers=2,
        mlp_ratio=4.0,
        dropout=0.0,
        training_type="keepall",
        random_mask_prob=0.0,
    ):
        super().__init__()
        if model_dim <= 0:
            raise ValueError(f"model_dim must be positive, got {model_dim}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})."
            )
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")

        self.modalities = tuple(input_dims.keys())
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.training_type = str(training_type)
        if self.training_type not in {"keepall", "mask"}:
            raise ValueError(
                "training_type must be either 'keepall' or 'mask'. "
                f"Got '{self.training_type}'."
            )
        self.supports_missing_modalities = self.training_type == "mask"
        self.random_mask_prob = float(random_mask_prob)
        if not 0.0 <= self.random_mask_prob < 1.0:
            raise ValueError(
                f"random_mask_prob must be in [0, 1). Got {self.random_mask_prob}."
            )
        self.projections = nn.ModuleDict(
            {
                name: nn.Linear(input_dim, model_dim)
                for name, input_dim in input_dims.items()
            }
        )
        self.modality_embeddings = nn.Parameter(
            torch.empty(len(self.modalities), model_dim)
        )
        self.mask_token = (
            nn.Parameter(torch.empty(1, 1, model_dim))
            if self.training_type == "mask"
            else None
        )
        self.state_token = nn.Parameter(torch.empty(1, 1, model_dim))

        ff_dim = int(model_dim * mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_proj = (
            nn.Identity()
            if output_dim == model_dim
            else nn.Linear(model_dim, output_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.modality_embeddings, std=0.02)
        if self.mask_token is not None:
            nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.state_token, std=0.02)

    def _sample_training_mask(self, available_mask):
        """
        Sample a random modality mask per batch element and broadcast it across time.
        available_mask: (B, T, M) boolean mask for modalities actually present in input.
        """
        if (
            (not self.training)
            or self.training_type != "mask"
            or self.random_mask_prob <= 0.0
        ):
            return torch.zeros_like(available_mask, dtype=torch.bool)

        batch_size, num_steps, num_modalities = available_mask.shape
        per_sequence_available = available_mask.any(dim=1, keepdim=True)
        random_mask = (
            torch.rand(batch_size, 1, num_modalities, device=available_mask.device)
            < self.random_mask_prob
        )
        random_mask = random_mask & per_sequence_available

        available_counts = per_sequence_available.sum(dim=-1)
        fully_masked = (random_mask.sum(dim=-1) >= available_counts) & (
            available_counts > 0
        )
        # If all available modalities for a batch element are masked out, randomly unmask one of them to ensure the model always has some modality signal during training.
        if fully_masked.any():
            for batch_idx in fully_masked.squeeze(-1).nonzero(as_tuple=False).flatten():
                available_indices = (
                    per_sequence_available[batch_idx, 0]
                    .nonzero(as_tuple=False)
                    .flatten()
                )
                keep_idx = available_indices[
                    torch.randint(
                        low=0,
                        high=len(available_indices),
                        size=(1,),
                        device=available_mask.device,
                    ).item()
                ]
                random_mask[batch_idx, 0, keep_idx] = False

        return random_mask.expand(-1, num_steps, -1)

    def forward(self, modality_embs):
        if not modality_embs:
            raise ValueError(
                "State-token attention fusion requires at least one available modality."
            )

        first_token = next(iter(modality_embs.values()))
        batch_size, num_steps = first_token.shape[:2]
        device = first_token.device
        dtype = first_token.dtype

        token_list = []
        available_masks = []
        # Find what are the available modalities and project them to tokens, while keeping track of which modalities are missing.
        for name in self.modalities:
            modality_value = modality_embs.get(name)
            if modality_value is None:
                token = torch.zeros(
                    batch_size,
                    num_steps,
                    self.model_dim,
                    device=device,
                    dtype=dtype,
                )
                available = torch.zeros(
                    batch_size, num_steps, device=device, dtype=torch.bool
                )
            else:
                token = self.projections[name](modality_value)
                available = torch.ones(
                    batch_size, num_steps, device=device, dtype=torch.bool
                )
            token_list.append(token)
            available_masks.append(available)

        tokens = torch.stack(token_list, dim=2)
        available_mask = torch.stack(available_masks, dim=2)

        # _sample_training_mask ensures that at least one modality is kept per batch element, so we don't need to worry about the case where all modalities are masked out.
        random_mask = self._sample_training_mask(available_mask)
        mask_positions = (~available_mask) | random_mask

        if self.supports_missing_modalities:
            # Replace missing modality tokens with a learned mask token
            tokens = torch.where(
                mask_positions.unsqueeze(-1),
                self.mask_token.to(device=device, dtype=tokens.dtype).expand(
                    batch_size, num_steps, len(self.modalities), self.model_dim
                ),
                tokens,
            )
        # Also for mask token we add the modality specific embeddings so that the model can learn to differentiate which modality is missing instead of just seeing a generic mask token. For modalities that are present, this just adds a learned bias.
        tokens = tokens + self.modality_embeddings.to(
            device=device, dtype=tokens.dtype
        ).view(1, 1, len(self.modalities), self.model_dim)
        batch_size, num_steps, num_modalities, token_dim = tokens.shape
        tokens = tokens.reshape(batch_size * num_steps, num_modalities, token_dim)

        state_token = self.state_token.to(device=device, dtype=tokens.dtype).expand(
            batch_size * num_steps, -1, -1
        )
        sequence = torch.cat([state_token, tokens], dim=1)
        sequence = self.encoder(sequence)

        fused = self.output_norm(sequence[:, 0])
        fused = self.output_proj(fused)
        fused = fused.reshape(batch_size, num_steps, -1)
        aux = {}
        if self.supports_missing_modalities:
            aux["masked_modality_fraction"] = mask_positions.float().mean().detach()
        return fused, aux


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
    fusion_type = str(cfg.type)
    ordered_input_dims = OrderedDict(input_dims)

    if fusion_type == "identity":
        return IdentityFusion(
            modalities=ordered_input_dims.keys(),
            output_dim=cfg.get("output_dim"),
        )

    if fusion_type == "concatproject":
        concat_cfg = cfg.get("concatproject", cfg)
        hidden_dim = concat_cfg.get(
            "hidden_dim",
            max(sum(ordered_input_dims.values()), cfg.output_dim),
        )
        return ConcatProjectFusion(
            input_dims=ordered_input_dims,
            output_dim=cfg.output_dim,
            hidden_dim=hidden_dim,
            dropout=concat_cfg.get("dropout", 0.0),
        )

    if fusion_type == "selfattention":
        selfattention_cfg = cfg.get("selfattention", cfg)
        return StateTokenAttentionFusion(
            input_dims=ordered_input_dims,
            output_dim=cfg.output_dim,
            model_dim=selfattention_cfg.get("model_dim", 128),
            num_heads=selfattention_cfg.get("num_heads", 4),
            num_layers=selfattention_cfg.get("num_layers", 2),
            mlp_ratio=selfattention_cfg.get("mlp_ratio", 4.0),
            dropout=selfattention_cfg.get("dropout", 0.0),
            training_type=selfattention_cfg.get("training_type", "keepall"),
            random_mask_prob=selfattention_cfg.get("random_mask_prob", 0.0),
        )

    if fusion_type == "cross_attention":
        return CrossAttentionFusion()

    if fusion_type == "moe":
        return MoEFusion()

    raise ValueError(f"Unsupported fusion type: {fusion_type}")
