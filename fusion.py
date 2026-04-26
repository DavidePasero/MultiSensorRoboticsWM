from collections import OrderedDict

import torch
from torch import nn


def _validate_random_mask_prob(random_mask_prob):
    random_mask_prob = float(random_mask_prob)
    if not 0.0 <= random_mask_prob < 1.0:
        raise ValueError(f"random_mask_prob must be in [0, 1). Got {random_mask_prob}.")

    return random_mask_prob


def _sample_training_mask(training, random_mask_prob, available_mask):
    """
    Sample a random modality mask per batch element and broadcast it across time.

    available_mask: (B, T, M) boolean mask for modalities actually present in input.
    """
    if (not training) or random_mask_prob <= 0.0:
        return torch.zeros_like(available_mask, dtype=torch.bool)

    batch_size, num_steps, num_modalities = available_mask.shape
    per_sequence_available = available_mask.any(dim=1, keepdim=True)
    random_mask = (
        torch.rand(batch_size, 1, num_modalities, device=available_mask.device)
        < random_mask_prob
    )
    random_mask = random_mask & per_sequence_available

    available_counts = per_sequence_available.sum(dim=-1)
    fully_masked = (random_mask.sum(dim=-1) >= available_counts) & (
        available_counts > 0
    )
    if fully_masked.any():
        for batch_idx in fully_masked.squeeze(-1).nonzero(as_tuple=False).flatten():
            available_indices = (
                per_sequence_available[batch_idx, 0].nonzero(as_tuple=False).flatten()
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


def _project_modality_tokens(modality_embs, modalities, projections, model_dim):
    if not modality_embs:
        raise ValueError("Fusion requires at least one available modality.")

    first_token = next(iter(modality_embs.values()))
    batch_size, num_steps = first_token.shape[:2]
    device = first_token.device
    dtype = first_token.dtype

    token_list = []
    available_masks = []
    for name in modalities:
        modality_value = modality_embs.get(name)
        if modality_value is None:
            token = torch.zeros(
                batch_size, num_steps, model_dim, device=device, dtype=dtype
            )
            available = torch.zeros(
                batch_size, num_steps, device=device, dtype=torch.bool
            )
        else:
            token = projections[name](modality_value)
            available = torch.ones(
                batch_size, num_steps, device=device, dtype=torch.bool
            )
        token_list.append(token)
        available_masks.append(available)

    tokens = torch.stack(token_list, dim=2)
    available_mask = torch.stack(available_masks, dim=2)
    return tokens, available_mask


def _apply_mask_token(tokens, available_mask, mask_token, training, random_mask_prob):
    random_mask = _sample_training_mask(training, random_mask_prob, available_mask)
    mask_positions = (~available_mask) | random_mask
    mask_tokens = mask_token.to(device=tokens.device, dtype=tokens.dtype).expand(
        tokens.shape[0], tokens.shape[1], tokens.shape[2], tokens.shape[3]
    )
    tokens = torch.where(mask_positions.unsqueeze(-1), mask_tokens, tokens)
    return tokens, mask_positions


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
    Concatenate modality embeddings and project them to the world-model latent.

    Every modality is first projected to a shared model dimension. The projected
    modality vectors are concatenated in a fixed order and passed through a
    projection MLP. Missing or randomly dropped modalities are replaced with a
    learned mask token in that shared space.
    """

    def __init__(
        self,
        input_dims,
        output_dim,
        hidden_dim=512,
        dropout=0.0,
        model_dim=128,
        random_mask_prob=0.0,
    ):
        super().__init__()
        self.modalities = tuple(input_dims.keys())
        self.model_dim = int(model_dim)
        self.random_mask_prob = _validate_random_mask_prob(random_mask_prob)
        self.supports_missing_modalities = True

        if self.model_dim <= 0:
            raise ValueError(f"model_dim must be positive, got {self.model_dim}.")

        self.projections = nn.ModuleDict(
            {
                name: nn.Linear(input_dim, self.model_dim)
                for name, input_dim in input_dims.items()
            }
        )
        self.mask_token = nn.Parameter(torch.empty(1, 1, self.model_dim))
        total_dim = len(self.modalities) * self.model_dim

        self.net = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, modality_embs):
        tokens, available_mask = _project_modality_tokens(
            modality_embs, self.modalities, self.projections, self.model_dim
        )
        tokens, mask_positions = _apply_mask_token(
            tokens,
            available_mask,
            self.mask_token,
            self.training,
            self.random_mask_prob,
        )

        fused = tokens.reshape(tokens.shape[0], tokens.shape[1], -1)
        aux = {"masked_modality_fraction": mask_positions.float().mean().detach()}
        return self.net(fused), aux


class FeaturewiseGatedFusion(BaseFusion):
    """
    GMU-inspired gated fusion with one gate network per modality.

    Each modality is projected to a shared model dimension. A modality-specific
    MLP then predicts feature-wise gate logits from the modality token and a
    global context token computed as the mean over all modality tokens. A
    softmax across modalities is applied independently for each feature
    dimension, and the fused token is the weighted sum of modality tokens.
    """

    supports_missing_modalities = True

    def __init__(
        self,
        input_dims,
        output_dim,
        model_dim=128,
        gate_hidden_dim=256,
        dropout=0.0,
        random_mask_prob=0.0,
    ):
        super().__init__()
        if model_dim <= 0:
            raise ValueError(f"model_dim must be positive, got {model_dim}.")
        if gate_hidden_dim <= 0:
            raise ValueError(
                f"gate_hidden_dim must be positive, got {gate_hidden_dim}."
            )

        self.modalities = tuple(input_dims.keys())
        self.model_dim = model_dim
        self.output_dim = output_dim
        self.random_mask_prob = _validate_random_mask_prob(random_mask_prob)
        self.projections = nn.ModuleDict(
            {
                name: nn.Linear(input_dim, model_dim)
                for name, input_dim in input_dims.items()
            }
        )
        self.gate_networks = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(2 * model_dim, gate_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(gate_hidden_dim, model_dim),
                )
                for name in self.modalities
            }
        )
        self.mask_token = nn.Parameter(torch.empty(1, 1, model_dim))
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_proj = (
            nn.Identity()
            if output_dim == model_dim
            else nn.Linear(model_dim, output_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, modality_embs):
        tokens, available_mask = _project_modality_tokens(
            modality_embs, self.modalities, self.projections, self.model_dim
        )
        tokens, mask_positions = _apply_mask_token(
            tokens,
            available_mask,
            self.mask_token,
            self.training,
            self.random_mask_prob,
        )
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
        return fused, {
            "masked_modality_fraction": mask_positions.float().mean().detach()
        }


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
        self.random_mask_prob = _validate_random_mask_prob(random_mask_prob)
        self.supports_missing_modalities = True
        self.projections = nn.ModuleDict(
            {
                name: nn.Linear(input_dim, model_dim)
                for name, input_dim in input_dims.items()
            }
        )
        self.modality_embeddings = nn.Parameter(
            torch.empty(len(self.modalities), model_dim)
        )
        self.mask_token = nn.Parameter(torch.empty(1, 1, model_dim))
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
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.state_token, std=0.02)

    def forward(self, modality_embs):
        tokens, available_mask = _project_modality_tokens(
            modality_embs, self.modalities, self.projections, self.model_dim
        )
        tokens, mask_positions = _apply_mask_token(
            tokens,
            available_mask,
            self.mask_token,
            self.training,
            self.random_mask_prob,
        )
        tokens = tokens + self.modality_embeddings.to(
            device=tokens.device, dtype=tokens.dtype
        ).view(1, 1, len(self.modalities), self.model_dim)
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
        return fused, {
            "masked_modality_fraction": mask_positions.float().mean().detach()
        }


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
        model_dim = concat_cfg.get("model_dim", 128)
        hidden_dim = concat_cfg.get(
            "hidden_dim",
            max(
                sum(ordered_input_dims.values()),
                cfg.output_dim,
                len(ordered_input_dims) * model_dim,
            ),
        )
        return ConcatProjectFusion(
            input_dims=ordered_input_dims,
            output_dim=cfg.output_dim,
            hidden_dim=hidden_dim,
            dropout=concat_cfg.get("dropout", 0.0),
            model_dim=model_dim,
            random_mask_prob=concat_cfg.get("random_mask_prob", 0.0),
        )

    if fusion_type == "gated":
        gated_cfg = cfg.get("gated", cfg)
        model_dim = gated_cfg.get("model_dim", 128)
        return FeaturewiseGatedFusion(
            input_dims=ordered_input_dims,
            output_dim=cfg.output_dim,
            model_dim=model_dim,
            gate_hidden_dim=gated_cfg.get("gate_hidden_dim", 2 * model_dim),
            dropout=gated_cfg.get("dropout", 0.0),
            random_mask_prob=gated_cfg.get("random_mask_prob", 0.0),
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
            random_mask_prob=selfattention_cfg.get("random_mask_prob", 0.0),
        )

    raise ValueError(f"Unsupported fusion type: {fusion_type}")
