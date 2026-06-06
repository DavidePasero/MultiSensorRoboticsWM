"""Missing-modality handling modules used before fusion."""

from collections import OrderedDict
from contextlib import nullcontext

import torch
from torch import nn


def _validate_ratio(name, value):
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1). Got {value}.")
    return value


def _sample_full_modality_mask(training, random_mask_prob, available_mask):
    """
    Sample whole-modality masks per batch element and broadcast them across time.

    available_mask: (B, T, M) boolean mask for modalities actually present in input.
    """
    if (not training) or random_mask_prob <= 0.0:
        return torch.zeros_like(available_mask, dtype=torch.bool)

    batch_size, _, num_modalities = available_mask.shape
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

    return random_mask.expand_as(available_mask)


def _stack_tokens(token_dict, modalities):
    return torch.stack([token_dict[name] for name in modalities], dim=2)


def _unstack_tokens(tokens, modalities):
    return OrderedDict(
        (name, tokens[:, :, idx]) for idx, name in enumerate(modalities)
    )


class BaseImputer(nn.Module):
    supports_missing_modalities = True

    def __init__(self, input_dims, model_dim, random_mask_prob=0.0):
        super().__init__()
        self.modalities = tuple(input_dims.keys())
        self.model_dim = int(model_dim)
        if self.model_dim <= 0:
            raise ValueError(f"model_dim must be positive, got {self.model_dim}.")

        self.random_mask_prob = _validate_ratio(
            "random_mask_prob", random_mask_prob
        )
        self.projections = nn.ModuleDict(
            {
                name: nn.Linear(input_dim, self.model_dim)
                for name, input_dim in input_dims.items()
            }
        )
        self.mask_tokens = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(1, 1, self.model_dim))
                for name in self.modalities
            }
        )
        self.reset_parameters()

    def reset_parameters(self):
        for token in self.mask_tokens.values():
            nn.init.normal_(token, std=0.02)

    def _project_tokens(
        self,
        modality_embs,
        *,
        detach_encoder_inputs=False,
        detach_projection=False,
    ):
        if not modality_embs:
            raise ValueError("Imputer requires at least one available modality.")

        first_token = next(iter(modality_embs.values()))
        batch_size, num_steps = first_token.shape[:2]
        device = first_token.device
        dtype = first_token.dtype

        token_dict = OrderedDict()
        available_masks = []
        for name in self.modalities:
            modality_value = modality_embs.get(name)
            if modality_value is None:
                token = torch.zeros(
                    batch_size, num_steps, self.model_dim, device=device, dtype=dtype
                )
                available = torch.zeros(
                    batch_size, num_steps, device=device, dtype=torch.bool
                )
            else:
                value = modality_value.detach() if detach_encoder_inputs else modality_value
                grad_ctx = torch.no_grad() if detach_projection else nullcontext()
                with grad_ctx:
                    token = self.projections[name](value)
                available = torch.ones(
                    batch_size, num_steps, device=device, dtype=torch.bool
                )

            token_dict[name] = token
            available_masks.append(available)

        available_mask = torch.stack(available_masks, dim=2)
        return token_dict, available_mask

    def _apply_missing_tokens(
        self,
        token_dict,
        full_mask,
        *,
        detach_mask_tokens=False,
    ):
        completed = OrderedDict()
        for idx, name in enumerate(self.modalities):
            token = token_dict[name]
            mask_token = self.mask_tokens[name]
            if detach_mask_tokens:
                mask_token = mask_token.detach()
            mask_token = mask_token.to(device=token.device, dtype=token.dtype).expand_as(
                token
            )
            completed[name] = torch.where(full_mask[:, :, idx].unsqueeze(-1), mask_token, token)
        return completed

    def _zero_loss(self, reference):
        return reference.new_zeros(())


class MissingTokenImputer(BaseImputer):
    def forward(self, modality_embs):
        token_dict, available_mask = self._project_tokens(modality_embs)
        random_mask = _sample_full_modality_mask(
            self.training, self.random_mask_prob, available_mask
        )
        full_mask = (~available_mask) | random_mask
        completed = self._apply_missing_tokens(token_dict, full_mask)
        reference = next(iter(completed.values()))
        return {
            "modality_tokens": completed,
            "imputer_recon_loss": self._zero_loss(reference),
            "imputer_missing_recon_loss": self._zero_loss(reference),
            "imputer_partial_recon_loss": self._zero_loss(reference),
            "masked_modality_fraction": full_mask.float().mean().detach(),
            "masked_feature_fraction": reference.new_zeros(()).detach(),
        }


class LatentReconstructionImputer(nn.Module):
    supports_missing_modalities = True

    def __init__(
        self,
        input_dims,
        model_dim=128,
        num_heads=4,
        num_layers=2,
        mlp_ratio=4.0,
        dropout=0.0,
        random_mask_prob=0.0,
    ):
        super().__init__()
        self.modalities = tuple(input_dims.keys())
        self.model_dim = int(model_dim)
        if self.model_dim <= 0:
            raise ValueError(f"model_dim must be positive, got {self.model_dim}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim ({self.model_dim}) must be divisible by num_heads ({num_heads})."
            )
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")

        self.random_mask_prob = _validate_ratio(
            "random_mask_prob", random_mask_prob
        )
        self.projections = nn.ModuleDict(
            {
                name: nn.Linear(input_dim, self.model_dim)
                for name, input_dim in input_dims.items()
            }
        )
        self.missing_token = nn.Parameter(torch.empty(1, 1, self.model_dim))
        self.modality_embeddings = nn.Parameter(
            torch.empty(len(self.modalities), self.model_dim)
        )

        ff_dim = int(self.model_dim * mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(self.model_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.missing_token, std=0.02)
        nn.init.normal_(self.modality_embeddings, std=0.02)

    def _project_tokens(self, modality_embs):
        if not modality_embs:
            raise ValueError("Imputer requires at least one available modality.")

        first_token = next(iter(modality_embs.values()))
        batch_size, num_steps = first_token.shape[:2]
        device = first_token.device
        dtype = first_token.dtype

        token_dict = OrderedDict()
        available_masks = []
        for name in self.modalities:
            modality_value = modality_embs.get(name)
            if modality_value is None:
                token = torch.zeros(
                    batch_size, num_steps, self.model_dim, device=device, dtype=dtype
                )
                available = torch.zeros(
                    batch_size, num_steps, device=device, dtype=torch.bool
                )
            else:
                token = self.projections[name](modality_value)
                available = torch.ones(
                    batch_size, num_steps, device=device, dtype=torch.bool
                )
            token_dict[name] = token
            available_masks.append(available)

        available_mask = torch.stack(available_masks, dim=2)
        return token_dict, available_mask

    def _build_transformer_inputs(self, token_dict, full_mask):
        tokens = []
        for idx, name in enumerate(self.modalities):
            token = token_dict[name]
            modality_embedding = self.modality_embeddings[idx].to(
                device=token.device, dtype=token.dtype
            ).view(1, 1, -1)
            missing_token = self.missing_token.to(
                device=token.device, dtype=token.dtype
            ).expand_as(token)
            token = torch.where(
                full_mask[:, :, idx].unsqueeze(-1),
                missing_token,
                token,
            )
            tokens.append(token + modality_embedding)
        return torch.stack(tokens, dim=2)

    def _predict_tokens(self, tokens):
        batch_size, num_steps, num_modalities, token_dim = tokens.shape
        tokens = tokens.reshape(batch_size * num_steps, num_modalities, token_dim)
        tokens = self.predictor(tokens)
        tokens = self.output_norm(tokens)
        tokens = tokens.reshape(batch_size, num_steps, num_modalities, token_dim)
        return _unstack_tokens(tokens, self.modalities)

    def _masked_mse(self, pred, target, mask):
        if not mask.any():
            return pred.new_zeros(())
        diff = (pred - target).pow(2)
        mask = mask.to(dtype=diff.dtype).expand_as(diff)
        return (diff * mask).sum() / mask.sum().clamp_min(1)

    def _zero_loss(self, reference):
        return reference.new_zeros(())

    def forward(self, modality_embs):
        token_dict, available_mask = self._project_tokens(modality_embs)
        random_mask = _sample_full_modality_mask(
            self.training, self.random_mask_prob, available_mask
        )
        full_mask = (~available_mask) | random_mask
        transformer_inputs = self._build_transformer_inputs(token_dict, full_mask)
        predicted_tokens = self._predict_tokens(transformer_inputs)

        reference = next(iter(predicted_tokens.values()))
        zero = self._zero_loss(reference)
        recon_loss = zero
        if self.training:
            pred_stack = _stack_tokens(predicted_tokens, self.modalities)
            target_stack = _stack_tokens(token_dict, self.modalities)
            recon_loss = self._masked_mse(
                pred_stack,
                target_stack,
                (random_mask & available_mask).unsqueeze(-1),
            )

        return {
            "modality_tokens": predicted_tokens,
            "imputer_recon_loss": recon_loss,
            "imputer_missing_recon_loss": recon_loss,
            "imputer_partial_recon_loss": zero,
            "masked_modality_fraction": full_mask.float().mean().detach(),
            "masked_feature_fraction": reference.new_zeros(()).detach(),
        }


class SelfMaskImputer(BaseImputer):
    def __init__(
        self,
        input_dims,
        model_dim=128,
        num_heads=4,
        num_layers=2,
        mlp_ratio=4.0,
        dropout=0.0,
        random_mask_prob=0.0,
        feature_mask_ratio=0.25,
        missing_weight=1.0,
        partial_weight=1.0,
    ):
        super().__init__(
            input_dims=input_dims,
            model_dim=model_dim,
            random_mask_prob=random_mask_prob,
        )
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if self.model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim ({self.model_dim}) must be divisible by num_heads ({num_heads})."
            )
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}.")

        self.feature_mask_ratio = _validate_ratio(
            "feature_mask_ratio", feature_mask_ratio
        )
        self.missing_weight = float(missing_weight)
        self.partial_weight = float(partial_weight)
        if self.missing_weight < 0.0:
            raise ValueError(
                f"missing_weight must be non-negative. Got {self.missing_weight}."
            )
        if self.partial_weight < 0.0:
            raise ValueError(
                f"partial_weight must be non-negative. Got {self.partial_weight}."
            )

        self.modality_embeddings = nn.Parameter(
            torch.empty(len(self.modalities), self.model_dim)
        )
        self.feature_mask_tokens = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(1, 1, self.model_dim))
                for name in self.modalities
            }
        )
        ff_dim = int(self.model_dim * mlp_ratio)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.predictor = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(self.model_dim)
        mask_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.mask_predictor = nn.TransformerEncoder(
            mask_encoder_layer, num_layers=num_layers
        )
        self.mask_output_norm = nn.LayerNorm(self.model_dim)
        self.mask_score_head = nn.Linear(self.model_dim, self.model_dim)

        self.reset_selfmask_parameters()

    def reset_selfmask_parameters(self):
        nn.init.normal_(self.modality_embeddings, std=0.02)
        for token in self.feature_mask_tokens.values():
            nn.init.normal_(token, std=0.02)

    def _apply_feature_mask(
        self,
        token_dict,
        feature_mask_values,
    ):
        masked = OrderedDict()
        for idx, name in enumerate(self.modalities):
            token = token_dict[name]
            mask_token = self.feature_mask_tokens[name]
            mask_token = mask_token.to(device=token.device, dtype=token.dtype).expand_as(
                token
            )
            mask_value = feature_mask_values[:, :, idx].to(dtype=token.dtype)
            masked[name] = token * (1.0 - mask_value) + mask_token * mask_value
        return masked

    def _add_modality_embeddings(self, token_dict):
        output = OrderedDict()
        for idx, name in enumerate(self.modalities):
            embedding = self.modality_embeddings[idx].to(
                device=token_dict[name].device, dtype=token_dict[name].dtype
            )
            output[name] = token_dict[name] + embedding.view(1, 1, -1)
        return output

    def _predict_tokens(self, token_dict):
        tokens = _stack_tokens(token_dict, self.modalities)
        batch_size, num_steps, num_modalities, token_dim = tokens.shape
        tokens = tokens.reshape(batch_size * num_steps, num_modalities, token_dim)
        tokens = self.predictor(tokens)
        tokens = self.output_norm(tokens)
        tokens = tokens.reshape(batch_size, num_steps, num_modalities, token_dim)
        return _unstack_tokens(tokens, self.modalities)

    def _predict_feature_mask(self, token_dict, full_mask, observed_kept):
        shape = (*observed_kept.shape, self.model_dim)
        if (not self.training) or self.feature_mask_ratio <= 0.0:
            zero = next(iter(token_dict.values())).new_zeros(shape)
            return torch.zeros(shape, device=zero.device, dtype=torch.bool), zero

        topk = max(1, int(round(self.model_dim * self.feature_mask_ratio)))
        topk = min(topk, self.model_dim)

        mask_inputs = self._apply_missing_tokens(token_dict, full_mask)
        mask_inputs = self._add_modality_embeddings(mask_inputs)
        tokens = _stack_tokens(mask_inputs, self.modalities)
        batch_size, num_steps, num_modalities, token_dim = tokens.shape
        tokens = tokens.reshape(batch_size * num_steps, num_modalities, token_dim)
        scores = self.mask_predictor(tokens)
        scores = self.mask_output_norm(scores)
        scores = self.mask_score_head(scores)
        scores = scores.reshape(batch_size, num_steps, num_modalities, token_dim)

        hard_mask = torch.zeros_like(scores, dtype=torch.bool)
        topk_idx = scores.topk(k=topk, dim=-1).indices
        hard_mask.scatter_(-1, topk_idx, True)
        hard_mask = hard_mask & observed_kept.unsqueeze(-1)

        soft_mask = torch.sigmoid(scores) * observed_kept.unsqueeze(-1).to(
            scores.dtype
        )
        straight_through_mask = (
            hard_mask.to(scores.dtype) + soft_mask - soft_mask.detach()
        )
        return hard_mask, straight_through_mask

    def _build_masked_inputs(
        self,
        token_dict,
        full_mask,
        feature_mask_values,
    ):
        token_dict = self._apply_feature_mask(
            token_dict,
            feature_mask_values,
        )
        token_dict = self._apply_missing_tokens(
            token_dict,
            full_mask,
        )
        return self._add_modality_embeddings(token_dict)

    def _masked_mse(self, pred, target, mask):
        if not mask.any():
            return pred.new_zeros(())
        diff = (pred - target).pow(2)
        mask = mask.to(dtype=diff.dtype).expand_as(diff)
        return (diff * mask).sum() / mask.sum().clamp_min(1)

    def forward(self, modality_embs):
        token_dict, available_mask = self._project_tokens(modality_embs)
        random_mask = _sample_full_modality_mask(
            self.training, self.random_mask_prob, available_mask
        )
        full_mask = (~available_mask) | random_mask
        observed_kept = available_mask & (~random_mask)
        feature_mask, feature_mask_values = self._predict_feature_mask(
            token_dict,
            full_mask,
            observed_kept,
        )

        jepa_inputs = self._build_masked_inputs(
            token_dict,
            full_mask,
            feature_mask_values,
        )
        predicted_tokens = self._predict_tokens(jepa_inputs)

        reference = next(iter(predicted_tokens.values()))
        zero = self._zero_loss(reference)
        missing_recon_loss = zero
        partial_recon_loss = zero
        recon_loss = zero

        if self.training:
            pred_stack = _stack_tokens(predicted_tokens, self.modalities)
            target_stack = _stack_tokens(token_dict, self.modalities)
            missing_recon_loss = self._masked_mse(
                pred_stack,
                target_stack,
                (random_mask & available_mask).unsqueeze(-1),
            )
            partial_recon_loss = self._masked_mse(
                pred_stack,
                target_stack,
                feature_mask,
            )
            recon_loss = (
                self.missing_weight * missing_recon_loss
                + self.partial_weight * partial_recon_loss
            )

        return {
            "modality_tokens": predicted_tokens,
            "imputer_recon_loss": recon_loss,
            "imputer_missing_recon_loss": missing_recon_loss,
            "imputer_partial_recon_loss": partial_recon_loss,
            "imputer_missing_weighted_loss": self.missing_weight * missing_recon_loss,
            "imputer_partial_weighted_loss": self.partial_weight * partial_recon_loss,
            "masked_modality_fraction": full_mask.float().mean().detach(),
            "masked_feature_fraction": feature_mask.float().mean().detach(),
        }


def build_imputer(obs_cfg, input_dims, default_model_dim):
    imputer_cfg = obs_cfg.get("imputer")
    if imputer_cfg is None:
        imputer_cfg = {
            "type": "missing_token",
            "model_dim": default_model_dim,
            "random_mask_prob": 0.0,
            "feature_mask_ratio": 0.25,
            "selfmask": {
                "num_heads": 4,
                "num_layers": 2,
                "mlp_ratio": 4.0,
                "dropout": 0.0,
                "missing_weight": 1.0,
                "partial_weight": 1.0,
            },
            "latent_reconstruction": {
                "num_heads": 4,
                "num_layers": 2,
                "mlp_ratio": 4.0,
                "dropout": 0.0,
            },
        }

    imputer_type = str(imputer_cfg.get("type", "missing_token"))
    model_dim = int(imputer_cfg.get("model_dim", default_model_dim))
    random_mask_prob = imputer_cfg.get("random_mask_prob", 0.0)

    if imputer_type == "missing_token":
        return MissingTokenImputer(
            input_dims=input_dims,
            model_dim=model_dim,
            random_mask_prob=random_mask_prob,
        )

    if imputer_type == "selfmask":
        selfmask_cfg = imputer_cfg.get("selfmask", imputer_cfg)
        return SelfMaskImputer(
            input_dims=input_dims,
            model_dim=model_dim,
            num_heads=selfmask_cfg.get("num_heads", 4),
            num_layers=selfmask_cfg.get("num_layers", 2),
            mlp_ratio=selfmask_cfg.get("mlp_ratio", 4.0),
            dropout=selfmask_cfg.get("dropout", 0.0),
            random_mask_prob=random_mask_prob,
            feature_mask_ratio=imputer_cfg.get("feature_mask_ratio", 0.25),
            missing_weight=selfmask_cfg.get("missing_weight", 1.0),
            partial_weight=selfmask_cfg.get("partial_weight", 1.0),
        )

    if imputer_type in {"latent_reconstruction", "latent_recon"}:
        latent_recon_cfg = imputer_cfg.get("latent_reconstruction", imputer_cfg)
        return LatentReconstructionImputer(
            input_dims=input_dims,
            model_dim=model_dim,
            num_heads=latent_recon_cfg.get("num_heads", 4),
            num_layers=latent_recon_cfg.get("num_layers", 2),
            mlp_ratio=latent_recon_cfg.get("mlp_ratio", 4.0),
            dropout=latent_recon_cfg.get("dropout", 0.0),
            random_mask_prob=random_mask_prob,
        )

    raise ValueError(f"Unsupported imputer type: {imputer_type}")
