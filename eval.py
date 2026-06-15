import os
import json
from collections import OrderedDict, defaultdict
from copy import deepcopy

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import time
from pathlib import Path

import h5py
import hydra
import numpy as np
import stable_pretraining as spt
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from multimodal import _apply_gaussian_blur
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def model_supports_missing_modalities(model):
    encoder = getattr(model, "encoder", None)
    imputer = getattr(encoder, "imputer", None) if encoder is not None else None
    if imputer is not None:
        return bool(getattr(imputer, "supports_missing_modalities", False))
    fusion = getattr(encoder, "fusion", None) if encoder is not None else None
    return bool(getattr(fusion, "supports_missing_modalities", False))


def model_uses_obs_encoder(model):
    encoder = getattr(model, "encoder", None)
    return bool(getattr(encoder, "is_obs_encoder", False))


def get_model_modality_sources(model):
    encoder = getattr(model, "encoder", None)
    encoders = getattr(encoder, "encoders", None)
    if encoders is None:
        return OrderedDict()

    sources = OrderedDict()
    for name, modality_encoder in encoders.items():
        sources[name] = getattr(modality_encoder, "source", name)
    return sources


def normalize_modalities_arg(value):
    if value is None:
        return []

    if isinstance(value, str):
        items = value.split(",")
    else:
        items = list(value)

    normalized = []
    for item in items:
        if item is None:
            continue
        item = str(item).strip()
        if item == "" or item.lower() == "none":
            continue
        if item not in normalized:
            normalized.append(item)
    return normalized


def get_drop_modalities(eval_cfg):
    requested = []
    requested.extend(normalize_modalities_arg(eval_cfg.get("drop_modality", None)))
    requested.extend(normalize_modalities_arg(eval_cfg.get("drop_modalities", [])))

    drop_modalities = []
    for modality in requested:
        if modality not in drop_modalities:
            drop_modalities.append(modality)
    return drop_modalities


def get_modality_substitution(eval_cfg):
    method = str(eval_cfg.get("modality_substitution", "impute")).strip().lower()
    aliases = {
        "drop": "impute",
        "missing": "impute",
        "mask": "impute",
        "zeros": "zero",
    }
    method = aliases.get(method, method)
    if method not in {"impute", "zero"}:
        raise ValueError(
            "eval.modality_substitution must be one of: impute, zero "
            f"(got {method!r})."
        )
    return method


def _make_json_safe(value):
    if isinstance(value, dict):
        return {k: _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


class EvalGaussianBlur:
    """Apply stochastic Gaussian blur once during policy input preparation."""

    def __init__(self, blur_cfg):
        self.blur_cfg = dict(blur_cfg)
        self.blur_cfg["enabled"] = True
        self.blur_cfg["training_only"] = False

    def __call__(self, image):
        blurred = _apply_gaussian_blur(
            torch.as_tensor(image).unsqueeze(0),
            self.blur_cfg,
            training=True,
        )
        return blurred.squeeze(0)


class ActionRegularizedCostModel(torch.nn.Module):
    """Add action priors to planning cost without changing model training."""

    def __init__(
        self,
        model,
        *,
        action_processor=None,
        action_norm_weight=0.0,
        action_delta_weight=0.0,
        first_action_delta_weight=0.0,
    ):
        super().__init__()
        self.model = model
        self.action_processor = action_processor
        self.action_norm_weight = float(action_norm_weight)
        self.action_delta_weight = float(action_delta_weight)
        self.first_action_delta_weight = float(first_action_delta_weight)

    @property
    def encoder(self):
        return self.model.encoder

    def encode(self, *args, **kwargs):
        return self.model.encode(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def rollout(self, *args, **kwargs):
        return self.model.rollout(*args, **kwargs)

    def criterion(self, *args, **kwargs):
        return self.model.criterion(*args, **kwargs)

    def _action_stats(self, actions):
        if self.action_processor is None:
            return None
        mean = getattr(self.action_processor, "mean_", None)
        scale = getattr(self.action_processor, "scale_", None)
        if mean is None or scale is None:
            return None

        action_dim = actions.size(-1)
        mean = torch.as_tensor(mean, dtype=actions.dtype, device=actions.device)
        scale = torch.as_tensor(scale, dtype=actions.dtype, device=actions.device)
        if mean.numel() != action_dim:
            if action_dim % mean.numel() != 0:
                return None
            repeat = action_dim // mean.numel()
            mean = mean.repeat(repeat)
            scale = scale.repeat(repeat)

        view_shape = [1] * actions.ndim
        view_shape[-1] = action_dim
        return mean.view(*view_shape), scale.view(*view_shape)

    def _to_env_action_units(self, actions):
        stats = self._action_stats(actions)
        if stats is None:
            return actions
        mean, scale = stats
        return actions * scale + mean

    def _last_history_action(self, history, num_samples, device):
        if not torch.is_tensor(history):
            return None
        history = history.to(device)

        if history.ndim == 4:
            last_action = history[..., -1, :]
        elif history.ndim == 3:
            last_action = history[:, -1, :].unsqueeze(1)
            last_action = last_action.expand(-1, num_samples, -1)
        elif history.ndim == 2:
            last_action = history.unsqueeze(1).expand(-1, num_samples, -1)
        else:
            return None

        return self._to_env_action_units(last_action)

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        history_action = info_dict.get("action")
        if torch.is_tensor(history_action):
            history_action = history_action.detach().clone()

        cost = self.model.get_cost(info_dict, action_candidates)
        if (
            self.action_norm_weight == 0.0
            and self.action_delta_weight == 0.0
            and self.first_action_delta_weight == 0.0
        ):
            return cost

        actions = self._to_env_action_units(action_candidates)
        penalty = torch.zeros_like(cost)

        if self.action_norm_weight != 0.0:
            action_norm_sq = actions.pow(2).sum(dim=-1).mean(dim=-1)
            penalty = penalty + self.action_norm_weight * action_norm_sq

        if self.action_delta_weight != 0.0 and actions.size(2) > 1:
            action_delta = actions[:, :, 1:] - actions[:, :, :-1]
            action_delta_sq = action_delta.pow(2).sum(dim=-1).mean(dim=-1)
            penalty = penalty + self.action_delta_weight * action_delta_sq

        if self.first_action_delta_weight != 0.0:
            last_action = self._last_history_action(
                history_action,
                actions.size(1),
                actions.device,
            )
            if last_action is not None:
                first_delta = actions[:, :, 0] - last_action
                first_delta_sq = first_delta.pow(2).sum(dim=-1)
                penalty = penalty + self.first_action_delta_weight * first_delta_sq

        return cost + penalty


class ActionClippedCEMSolver:
    """CEM wrapper that optimizes only valid actions in model action units."""

    def __init__(
        self,
        base_solver,
        action_processor=None,
        first_action_delta_limit=None,
        action_delta_limit=None,
    ):
        self.base_solver = base_solver
        self.action_processor = action_processor
        self.first_action_delta_limit = first_action_delta_limit
        self.action_delta_limit = action_delta_limit
        self._candidate_low = None
        self._candidate_high = None

    def configure(self, *, action_space, n_envs, config) -> None:
        self.base_solver.configure(
            action_space=action_space,
            n_envs=n_envs,
            config=config,
        )
        low = np.asarray(action_space.low, dtype=np.float32).reshape(n_envs, -1)
        high = np.asarray(action_space.high, dtype=np.float32).reshape(n_envs, -1)

        if self.action_processor is not None:
            low = self.action_processor.transform(low)
            high = self.action_processor.transform(high)

        low, high = np.minimum(low, high), np.maximum(low, high)
        action_block = int(getattr(config, "action_block", 1))
        if action_block > 1:
            low = np.repeat(low, action_block, axis=1)
            high = np.repeat(high, action_block, axis=1)

        self._candidate_low = torch.as_tensor(low, dtype=torch.float32)
        self._candidate_high = torch.as_tensor(high, dtype=torch.float32)

    @property
    def n_envs(self):
        return self.base_solver.n_envs

    @property
    def action_dim(self):
        return self.base_solver.action_dim

    @property
    def horizon(self):
        return self.base_solver.horizon

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def _clip_candidates(self, candidates, start_idx=0, end_idx=None):
        if self._candidate_low is None or self._candidate_high is None:
            return candidates
        end_idx = end_idx if end_idx is not None else start_idx + candidates.size(0)
        low = self._candidate_low[start_idx:end_idx].to(candidates.device)
        high = self._candidate_high[start_idx:end_idx].to(candidates.device)
        return torch.clamp(candidates, low[:, None, None, :], high[:, None, None, :])

    def _delta_limit_in_model_units(self, limit_value, action_dim, *, dtype, device):
        if limit_value is None:
            return None

        limit = float(limit_value)
        if limit <= 0.0:
            return None

        delta_limit = torch.full((action_dim,), limit, dtype=dtype, device=device)
        if self.action_processor is None:
            return delta_limit

        scale = getattr(self.action_processor, "scale_", None)
        if scale is None:
            return delta_limit

        scale = torch.as_tensor(scale, dtype=dtype, device=device).clamp_min(1e-6)
        if scale.numel() != action_dim:
            if action_dim % scale.numel() != 0:
                return delta_limit
            scale = scale.repeat(action_dim // scale.numel())
        return delta_limit / scale

    def _last_history_action(self, info_dict, start_idx, end_idx, *, device):
        history = info_dict.get("action")
        if history is None:
            return None
        if isinstance(history, np.ndarray):
            history = torch.as_tensor(history)
        if not torch.is_tensor(history):
            return None

        history = history[start_idx:end_idx].to(device)
        if history.ndim == 4:
            history = history[:, 0]
        if history.ndim == 3:
            return history[:, -1, :]
        if history.ndim == 2:
            return history
        return None

    def _limit_first_action_delta(self, candidates, last_action):
        limit = self._delta_limit_in_model_units(
            self.first_action_delta_limit,
            candidates.size(-1),
            dtype=candidates.dtype,
            device=candidates.device,
        )
        if limit is None or last_action is None:
            return candidates

        if last_action.size(-1) != candidates.size(-1):
            if candidates.size(-1) % last_action.size(-1) != 0:
                return candidates
            last_action = last_action.repeat(1, candidates.size(-1) // last_action.size(-1))

        low = last_action[:, None, :] - limit.view(1, 1, -1)
        high = last_action[:, None, :] + limit.view(1, 1, -1)
        candidates[:, :, 0, :] = torch.clamp(candidates[:, :, 0, :], low, high)
        return candidates

    def _limit_action_deltas(self, candidates, last_action):
        limit = self._delta_limit_in_model_units(
            self.action_delta_limit,
            candidates.size(-1),
            dtype=candidates.dtype,
            device=candidates.device,
        )
        if limit is None or last_action is None:
            return candidates

        if last_action.size(-1) != candidates.size(-1):
            if candidates.size(-1) % last_action.size(-1) != 0:
                return candidates
            last_action = last_action.repeat(
                1,
                candidates.size(-1) // last_action.size(-1),
            )

        prev = last_action[:, None, :]
        limit = limit.view(1, 1, -1)
        for t in range(candidates.size(2)):
            candidates[:, :, t, :] = torch.clamp(
                candidates[:, :, t, :],
                prev - limit,
                prev + limit,
            )
            prev = candidates[:, :, t, :]
        return candidates

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        start_time = time.time()
        solver = self.base_solver
        outputs = {"costs": [], "mean": [], "var": []}

        mean, var = solver.init_action_distrib(init_action)
        mean = mean.to(solver.device)
        var = var.to(solver.device)
        mean = self._clip_candidates(mean.unsqueeze(1)).squeeze(1)

        total_envs = solver.n_envs
        for start_idx in range(0, total_envs, solver.batch_size):
            end_idx = min(start_idx + solver.batch_size, total_envs)
            current_bs = end_idx - start_idx
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]
            last_history_action = self._last_history_action(
                info_dict,
                start_idx,
                end_idx,
                device=solver.device,
            )

            expanded_infos = {}
            for key, value in info_dict.items():
                value_batch = value[start_idx:end_idx]
                if torch.is_tensor(value):
                    value_batch = value_batch.unsqueeze(1)
                    value_batch = value_batch.expand(
                        current_bs,
                        solver.num_samples,
                        *value_batch.shape[2:],
                    )
                elif isinstance(value, np.ndarray):
                    value_batch = np.repeat(
                        value_batch[:, None, ...],
                        solver.num_samples,
                        axis=1,
                    )
                expanded_infos[key] = value_batch

            final_batch_cost = None
            for _ in range(solver.n_steps):
                candidates = torch.randn(
                    current_bs,
                    solver.num_samples,
                    solver.horizon,
                    solver.action_dim,
                    generator=solver.torch_gen,
                    device=solver.device,
                )
                candidates = (
                    candidates * batch_var.unsqueeze(1)
                    + batch_mean.unsqueeze(1)
                )
                candidates[:, 0] = batch_mean
                candidates = self._clip_candidates(candidates, start_idx, end_idx)
                candidates = self._limit_first_action_delta(
                    candidates,
                    last_history_action,
                )
                candidates = self._limit_action_deltas(
                    candidates,
                    last_history_action,
                )

                costs = solver.model.get_cost(expanded_infos.copy(), candidates)
                if not isinstance(costs, torch.Tensor):
                    raise AssertionError(
                        f"Expected cost to be a torch.Tensor, got {type(costs)}"
                    )
                if costs.ndim != 2 or costs.shape != (
                    current_bs,
                    solver.num_samples,
                ):
                    raise AssertionError(
                        "Expected cost shape "
                        f"({current_bs}, {solver.num_samples}), got {costs.shape}"
                    )

                topk_vals, topk_inds = torch.topk(
                    costs,
                    k=solver.topk,
                    dim=1,
                    largest=False,
                )
                batch_indices = torch.arange(
                    current_bs,
                    device=solver.device,
                ).unsqueeze(1).expand(-1, solver.topk)
                topk_candidates = candidates[batch_indices, topk_inds]
                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1)
                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_batch_cost)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        print(f"CEM solve time: {time.time() - start_time:.4f} seconds")
        return outputs


def get_eval_pixels_gaussian_blur(eval_cfg):
    blur_cfg = eval_cfg.get("pixels_gaussian_blur", None)
    if blur_cfg is None or not blur_cfg.get("enabled", False):
        return None

    blur_cfg = OmegaConf.to_container(blur_cfg, resolve=True)
    blur_cfg["enabled"] = True
    return blur_cfg


class ModalityDropoutWorldModelPolicy(swm.policy.WorldModelPolicy):
    """World-model policy that removes a modality so missing-modality fusion can mask it."""

    def __init__(
        self,
        *args,
        drop_modalities=None,
        modality_substitution="impute",
        execution_action_delta_limit=None,
        execution_action_norm_limit=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.drop_modalities = normalize_modalities_arg(drop_modalities)
        self.modality_substitution = str(modality_substitution)
        self.execution_action_delta_limit = execution_action_delta_limit
        self.execution_action_norm_limit = execution_action_norm_limit
        self.executed_actions = []
        self.pre_limit_executed_actions = []
        self.execution_action_limited = []
        self.solver_costs = []
        self._last_executed_action = None

    def _resolve_drop_sources(self, strict=True):
        if not self.drop_modalities:
            return []

        model = getattr(self.solver, "model", None)
        modality_sources = get_model_modality_sources(model)
        if not modality_sources:
            return list(self.drop_modalities)

        known = sorted(set(modality_sources) | set(modality_sources.values()))
        resolved_sources = []
        for modality in self.drop_modalities:
            if modality in modality_sources:
                source = modality_sources[modality]
            elif modality in modality_sources.values():
                source = modality
            else:
                if not strict:
                    continue
                raise ValueError(
                    f"Unknown modality '{modality}'. Available modalities: {known}."
                )

            if source not in resolved_sources:
                resolved_sources.append(source)
        return resolved_sources

    def _find_fallback_primary_source(self, info_dict, dropped_sources):
        model = getattr(self.solver, "model", None)
        modality_sources = get_model_modality_sources(model)
        for source in modality_sources.values():
            if source in dropped_sources or source not in info_dict:
                continue

            goal_key = "goal" if source == "pixels" else f"goal_{source}"
            if goal_key in info_dict:
                return source
        return None

    def _drop_selected_modalities(self, info_dict):
        drop_sources = self._resolve_drop_sources()
        if not drop_sources:
            return info_dict, None, None

        pruned = dict(info_dict)
        for drop_source in drop_sources:
            pruned.pop(drop_source, None)
            pruned.pop(
                "goal" if drop_source == "pixels" else f"goal_{drop_source}",
                None,
            )

        model = getattr(self.solver, "model", None)
        encoder = getattr(model, "encoder", None)
        original_primary_source = getattr(encoder, "primary_source", None)

        if encoder is not None and original_primary_source in drop_sources:
            fallback_source = self._find_fallback_primary_source(pruned, drop_sources)
            if fallback_source is None:
                raise ValueError(
                    "Cannot drop the requested modalities because no fallback "
                    "goal-conditioned modality is available."
                )

            fallback_goal_key = (
                "goal" if fallback_source == "pixels" else f"goal_{fallback_source}"
            )
            encoder.primary_source = fallback_source
            pruned["goal"] = pruned[fallback_goal_key]

        return pruned, encoder, original_primary_source

    def _zero_selected_modalities(self, info_dict):
        drop_sources = self._resolve_drop_sources(strict=False)
        if not drop_sources:
            return info_dict, None, None

        zeroed = dict(info_dict)
        for drop_source in drop_sources:
            for key in (
                drop_source,
                "goal" if drop_source == "pixels" else f"goal_{drop_source}",
            ):
                value = zeroed.get(key)
                if torch.is_tensor(value):
                    zeroed[key] = torch.zeros_like(value)
                elif isinstance(value, np.ndarray):
                    zeroed[key] = np.zeros_like(value)

        return zeroed, None, None

    def _substitute_selected_modalities(self, info_dict):
        if self.modality_substitution == "zero":
            return self._zero_selected_modalities(info_dict)
        return self._drop_selected_modalities(info_dict)

    def _previous_action_from_info(self, info_dict, action_shape):
        if self._last_executed_action is not None:
            return self._last_executed_action

        history = info_dict.get("action")
        if history is None:
            return np.zeros(action_shape, dtype=np.float32)

        history = np.asarray(history, dtype=np.float32)
        if history.ndim >= 3:
            previous = history[:, -1, :]
        elif history.ndim == 2:
            previous = history
        elif history.ndim == 1:
            previous = history.reshape(1, -1)
        else:
            return np.zeros(action_shape, dtype=np.float32)

        try:
            return np.broadcast_to(previous, action_shape).copy()
        except ValueError:
            return np.zeros(action_shape, dtype=np.float32)

    def _limit_executed_action_delta(self, action, raw_info_dict):
        if self.execution_action_delta_limit is None:
            return action, False

        limit = float(self.execution_action_delta_limit)
        if limit <= 0.0:
            return action, False

        previous = self._previous_action_from_info(raw_info_dict, action.shape)
        limited = np.clip(action, previous - limit, previous + limit)

        action_space = getattr(self.env, "action_space", None)
        if action_space is not None:
            low = np.asarray(action_space.low, dtype=np.float32)
            high = np.asarray(action_space.high, dtype=np.float32)
            limited = np.clip(limited, low, high)

        changed = not np.allclose(action, limited)
        return limited.astype(action.dtype, copy=False), changed

    def _limit_executed_action_norm(self, action):
        if self.execution_action_norm_limit is None:
            return action, False

        limit = float(self.execution_action_norm_limit)
        if limit <= 0.0:
            return action, False

        norms = np.linalg.norm(action, axis=-1, keepdims=True)
        scale = np.minimum(1.0, limit / np.maximum(norms, 1e-8))
        limited = action * scale

        action_space = getattr(self.env, "action_space", None)
        if action_space is not None:
            low = np.asarray(action_space.low, dtype=np.float32)
            high = np.asarray(action_space.high, dtype=np.float32)
            limited = np.clip(limited, low, high)

        changed = not np.allclose(action, limited)
        return limited.astype(action.dtype, copy=False), changed

    def _limit_executed_action(self, action, raw_info_dict):
        action, delta_changed = self._limit_executed_action_delta(
            action,
            raw_info_dict,
        )
        action, norm_changed = self._limit_executed_action_norm(action)
        self.execution_action_limited.append(delta_changed or norm_changed)
        return action

    def get_action(self, info_dict: dict, **kwargs):
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"
        raw_info_dict = dict(info_dict)

        prepared_info = self._prepare_info(dict(info_dict))
        prepared_info, encoder, original_primary_source = (
            self._substitute_selected_modalities(prepared_info)
        )

        try:
            if len(self._action_buffer) == 0:
                outputs = self.solver(prepared_info, init_action=self._next_init)
                if "costs" in outputs:
                    self.solver_costs.extend(outputs["costs"])

                actions = outputs["actions"]
                keep_horizon = self.cfg.receding_horizon
                plan = actions[:, :keep_horizon]
                rest = actions[:, keep_horizon:]
                self._next_init = rest if self.cfg.warm_start else None

                plan = plan.reshape(
                    self.env.num_envs, self.flatten_receding_horizon, -1
                )
                self._action_buffer.extend(plan.transpose(0, 1))
        finally:
            if encoder is not None and original_primary_source is not None:
                encoder.primary_source = original_primary_source

        action = self._action_buffer.popleft()
        action = action.reshape(*self.env.action_space.shape)
        action = action.numpy()

        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)

        self.pre_limit_executed_actions.append(action.copy())
        action = self._limit_executed_action(action, raw_info_dict)
        self.executed_actions.append(action.copy())
        self._last_executed_action = action.copy()
        return action

    def prepare_eval_info(self, info_dict: dict):
        prepared_info = self._prepare_info(dict(info_dict))
        prepared_info, encoder, original_primary_source = (
            self._substitute_selected_modalities(prepared_info)
        )
        return prepared_info, encoder, original_primary_source


def img_transform(cfg):
    blur_cfg = get_eval_pixels_gaussian_blur(cfg.eval)
    if blur_cfg is None:
        return transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(**spt.data.dataset_stats.ImageNet),
                transforms.Resize(size=cfg.eval.img_size),
            ]
        )

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Resize(size=cfg.eval.img_size),
            EvalGaussianBlur(blur_cfg),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        ]
    )


def obs_encoder_eval_img_transform(cfg):
    blur_cfg = get_eval_pixels_gaussian_blur(cfg.eval)
    if blur_cfg is None:
        return None

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            EvalGaussianBlur(blur_cfg),
        ]
    )


def build_eval_image_transform(cfg, model):
    if not model_uses_obs_encoder(model):
        image_transform = img_transform(cfg)
        return {
            "pixels": image_transform,
            "goal": image_transform,
        }

    image_transform = obs_encoder_eval_img_transform(cfg)
    if image_transform is None:
        print("Using raw image observations; model obs encoder handles preprocessing.")
        return {}

    print("Applying eval pixel blur before model obs-encoder preprocessing.")
    return {
        "pixels": image_transform,
        "goal": image_transform,
    }


def resolve_dataset_env_idx(dataset, cfg):
    """Return the dataset env_idx matching cfg.world.metaworld_env_name, if possible."""
    explicit = cfg.eval.get("env_idx", None)
    if explicit is not None:
        return int(explicit)

    env_name = cfg.world.get("metaworld_env_name", None)
    if env_name is None or "env_idx" not in dataset.column_names:
        return None

    h5_path = getattr(dataset, "h5_path", None)
    if h5_path is None:
        return None

    with h5py.File(h5_path, "r") as f:
        names_json = f.attrs.get("env_names_json", None)
    if names_json is None:
        return None

    env_names = json.loads(names_json)
    if env_name not in env_names:
        raise ValueError(
            f"Requested MetaWorld env {env_name!r} is not in dataset env_names_json. "
            f"Available examples: {env_names[:5]} ..."
        )
    return env_names.index(env_name)


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def dataset_has_key(dataset, key: str) -> bool:
    h5_path = getattr(dataset, "h5_path", None)
    if h5_path is None:
        return key in dataset.column_names
    with h5py.File(h5_path, "r") as f:
        return key in f


def success_mask(dataset, success_key: str) -> np.ndarray:
    values = np.asarray(dataset.get_col_data(success_key))
    return values.reshape(values.shape[0], -1).any(axis=1)


class DatasetColumnFilter:
    """Hide bookkeeping columns from evaluate_from_dataset."""

    def __init__(self, dataset, hidden_keys):
        self.dataset = dataset
        self.hidden_keys = set(hidden_keys)

    @property
    def column_names(self):
        return [key for key in self.dataset.column_names if key not in self.hidden_keys]

    def get_col_data(self, col: str):
        return self.dataset.get_col_data(col)

    def get_row_data(self, row_idx):
        return self.dataset.get_row_data(row_idx)

    def load_chunk(self, episodes_idx, start, end):
        chunks = self.dataset.load_chunk(episodes_idx, start, end)
        for chunk in chunks:
            for key in self.hidden_keys:
                chunk.pop(key, None)
        return chunks


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _find_stacked_wrapper(env):
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if current.__class__.__name__ == "StackedWrapper" and hasattr(
            current, "buffers"
        ):
            return current
        current = getattr(current, "env", None)
    return None


def _set_stacked_history_buffers(world, init_history):
    """Synchronize StableWM's internal frame-stack buffers with dataset history."""
    for env_idx, env in enumerate(world.envs.unwrapped.envs):
        stacked = _find_stacked_wrapper(env)
        if stacked is None:
            continue

        for key, values in init_history.items():
            if key not in stacked.buffers:
                continue
            buffer = stacked.buffers[key]
            buffer.clear()
            for value in values[env_idx]:
                buffer.append(deepcopy(value))


def _collect_state_restore_metrics(world, current_step):
    metrics = {}
    qpos_errors = []
    proprio_errors = []
    mocap_hand_distances = []

    for env_idx, env in enumerate(world.envs.unwrapped.envs):
        env_unwrapped = env.unwrapped
        metaworld_env = getattr(env_unwrapped, "metaworld_env", None)
        if metaworld_env is None:
            continue

        inner = metaworld_env.unwrapped
        data = inner.data
        model = inner.model

        if "qpos" in current_step:
            expected_qpos = np.asarray(current_step["qpos"][env_idx]).reshape(-1)
            actual_qpos = np.asarray(data.qpos).reshape(-1)
            dim = min(expected_qpos.size, actual_qpos.size)
            if dim:
                qpos_errors.append(float(np.linalg.norm(actual_qpos[:dim] - expected_qpos[:dim])))

        if "proprio" in current_step:
            expected_proprio = np.asarray(current_step["proprio"][env_idx]).reshape(-1)
            actual_proprio = np.asarray(data.qpos[:7]).reshape(-1)
            dim = min(expected_proprio.size, actual_proprio.size)
            if dim:
                proprio_errors.append(
                    float(np.linalg.norm(actual_proprio[:dim] - expected_proprio[:dim]))
                )

        try:
            mocap_id = model.body_mocapid[data.body("mocap").id]
            if mocap_id >= 0:
                mocap_pos = np.asarray(data.mocap_pos[mocap_id]).reshape(-1)
                hand_pos = np.asarray(inner.get_endeff_pos()).reshape(-1)
                mocap_hand_distances.append(float(np.linalg.norm(mocap_pos - hand_pos)))
        except Exception:
            pass

    def add_stats(prefix, values):
        if not values:
            return
        arr = np.asarray(values, dtype=np.float32)
        metrics[f"{prefix}_mean"] = float(arr.mean())
        metrics[f"{prefix}_max"] = float(arr.max())

    add_stats("state_restore_qpos_l2", qpos_errors)
    add_stats("state_restore_proprio_l2", proprio_errors)
    add_stats("state_restore_mocap_hand_l2", mocap_hand_distances)
    return metrics


def _successes_from_world(world):
    if "success" in world.infos:
        successes = np.asarray(world.infos["success"])
        if successes.ndim > 1:
            successes = successes.reshape(successes.shape[0], -1).any(axis=1)
        return successes.astype(bool)
    return np.asarray(world.terminateds, dtype=bool)


def _add_policy_debug_metrics(results, policy, world):
    actions = getattr(policy, "executed_actions", None)
    if actions:
        action_arr = np.asarray(actions, dtype=np.float32)
        action_norms = np.linalg.norm(action_arr, axis=-1)
        results["action_norm_mean"] = float(action_norms.mean())
        results["action_norm_max"] = float(action_norms.max())
        results["first_action_norm_mean"] = float(action_norms[0].mean())
        results["first_action_norm_max"] = float(action_norms[0].max())

        action_space = getattr(world, "action_space", None)
        if action_space is not None:
            low = np.asarray(action_space.low, dtype=np.float32)
            high = np.asarray(action_space.high, dtype=np.float32)
            bound = np.maximum(np.abs(low), np.abs(high))
            while bound.ndim < action_arr.ndim:
                bound = np.expand_dims(bound, axis=0)
            bound = np.broadcast_to(bound, action_arr.shape)
            valid_bound = bound > 0
            saturated = np.zeros_like(action_arr, dtype=bool)
            saturated[valid_bound] = (
                np.abs(action_arr[valid_bound]) >= 0.95 * bound[valid_bound]
            )
            results["action_saturation_fraction"] = float(saturated.mean())

        if action_arr.shape[0] > 1:
            action_deltas = np.diff(action_arr, axis=0)
            action_delta_norms = np.linalg.norm(action_deltas, axis=-1)
            results["action_delta_norm_mean"] = float(action_delta_norms.mean())
            results["action_delta_norm_max"] = float(action_delta_norms.max())

        pre_limit_actions = getattr(policy, "pre_limit_executed_actions", None)
        if pre_limit_actions:
            pre_limit_arr = np.asarray(pre_limit_actions, dtype=np.float32)
            pre_limit_norms = np.linalg.norm(pre_limit_arr, axis=-1)
            results["pre_limit_action_norm_mean"] = float(pre_limit_norms.mean())
            results["pre_limit_action_norm_max"] = float(pre_limit_norms.max())
            first_delta = pre_limit_arr[0] - action_arr[0]
            first_delta_norms = np.linalg.norm(first_delta, axis=-1)
            results["first_action_execution_clip_delta_mean"] = float(
                first_delta_norms.mean()
            )
            results["first_action_execution_clip_delta_max"] = float(
                first_delta_norms.max()
            )

        limited = getattr(policy, "execution_action_limited", None)
        if limited:
            results["execution_action_limited_fraction"] = float(np.mean(limited))

    costs = getattr(policy, "solver_costs", None)
    if costs:
        cost_arr = np.asarray(costs, dtype=np.float32)
        results["plan_cost_mean"] = float(cost_arr.mean())
        results["plan_cost_std"] = float(cost_arr.std())


def _build_action_history(values, history_size):
    """Build live-env-style action history ending before the planning action."""
    if torch.is_tensor(values):
        action_history = values[:history_size].clone()
        action_history[0] = values[0]
        if history_size > 1:
            action_history[1:] = values[: history_size - 1]
        return action_history

    values = np.asarray(values)
    action_history = values[:history_size].copy()
    action_history[0] = values[0]
    if history_size > 1:
        action_history[1:] = values[: history_size - 1]
    return action_history


def evaluate_from_dataset_with_history(
    world,
    dataset,
    episodes_idx,
    start_steps,
    goal_offset_steps,
    eval_budget,
    callables=None,
    save_video=True,
    video_path="./",
):
    """Evaluate from dataset starts while preserving a true observation history.

    StableWM's built-in evaluate_from_dataset repeats the planning start frame
    across the history dimension. That is fine for history_size=1, but it
    creates a training/evaluation mismatch for LeMuMoWM checkpoints trained with
    a multi-step context. This variant loads the H-1 rows before each planning
    start and initializes both world.infos and the internal StackedWrapper
    buffers with those dataset histories.
    """
    history_size = int(getattr(world, "_history_size", 1))
    if history_size <= 1:
        print("Using default dataset evaluation with history_size=1.")
        return world.evaluate_from_dataset(
            dataset,
            start_steps=start_steps,
            goal_offset_steps=goal_offset_steps,
            eval_budget=eval_budget,
            episodes_idx=episodes_idx,
            callables=callables,
            save_video=save_video,
            video_path=video_path,
        )

    if (
        world.envs.envs[0].spec.max_episode_steps is not None
        and world.envs.envs[0].spec.max_episode_steps < goal_offset_steps
    ):
        raise AssertionError("env max_episode_steps must be greater than eval_budget")

    ep_idx_arr = np.asarray(episodes_idx)
    start_steps_arr = np.asarray(start_steps)
    end_steps = start_steps_arr + goal_offset_steps
    history_starts = start_steps_arr - (history_size - 1)

    if len(ep_idx_arr) != len(start_steps_arr):
        raise ValueError("episodes_idx and start_steps must have the same length")
    if len(ep_idx_arr) != world.num_envs:
        raise ValueError("Number of episodes to evaluate must match number of envs")
    if np.any(history_starts < 0):
        bad = np.nonzero(history_starts < 0)[0].tolist()
        raise ValueError(
            "Cannot build the requested evaluation history because some planning "
            f"starts have fewer than {history_size - 1} previous steps: {bad}."
        )
    print(
        "Using history-aware dataset evaluation with "
        f"history_size={history_size}; planning starts "
        f"{int(start_steps_arr.min())}..{int(start_steps_arr.max())}, "
        f"history starts {int(history_starts.min())}..{int(history_starts.max())}."
    )

    data = dataset.load_chunk(ep_idx_arr, history_starts, end_steps)
    columns = dataset.column_names

    init_history_per_env = defaultdict(list)
    current_step_per_env = defaultdict(list)
    goal_step_per_env = defaultdict(list)
    target_frame_chunks = []

    for ep in data:
        for col in columns:
            if col.startswith("goal") or col not in ep:
                continue
            if col.startswith("pixels"):
                ep[col] = ep[col].permute(0, 2, 3, 1)

            if not isinstance(ep[col], (torch.Tensor, np.ndarray)):
                continue
            if len(ep[col]) < history_size:
                raise ValueError(
                    f"Loaded chunk for column {col!r} is shorter than "
                    f"history_size={history_size}."
                )

            if col == "action":
                init_data = _build_action_history(ep[col], history_size)
            else:
                init_data = ep[col][:history_size]

            current_data = ep[col][history_size - 1]
            goal_data = ep[col][-1]

            init_history_per_env[col].append(_to_numpy(init_data))
            current_step_per_env[col].append(_to_numpy(current_data))
            goal_step_per_env[col].append(_to_numpy(goal_data))

        if "pixels" in ep:
            target_frame_chunks.append(_to_numpy(ep["pixels"][history_size - 1 :]))

    init_history = {
        key: np.stack(value) for key, value in deepcopy(init_history_per_env).items()
    }
    current_step = {
        key: np.stack(value) for key, value in deepcopy(current_step_per_env).items()
    }

    goal_step_single = {}
    for key, value in goal_step_per_env.items():
        goal_key = "goal" if key == "pixels" else f"goal_{key}"
        goal_step_single[goal_key] = np.stack(value)

    seeds = current_step.get("seed")
    variation_prefix = "variation."
    variations_dict = {
        key.removeprefix(variation_prefix): value
        for key, value in current_step.items()
        if key.startswith(variation_prefix)
    }

    options = [{} for _ in range(world.num_envs)]
    if variations_dict:
        for idx in range(world.num_envs):
            options[idx]["variation"] = list(variations_dict.keys())
            options[idx]["variation_values"] = {
                key: value[idx] for key, value in variations_dict.items()
            }

    callable_data = deepcopy(current_step)
    callable_data.update(deepcopy(goal_step_single))
    world.reset(seed=seeds, options=options)

    callables = callables or []
    for env_idx, env in enumerate(world.envs.unwrapped.envs):
        env_unwrapped = env.unwrapped
        for spec in callables:
            method_name = spec["method"]
            if not hasattr(env_unwrapped, method_name):
                print(
                    f"Env {env_unwrapped} has no method {method_name}, "
                    "skipping callable."
                )
                continue

            method = getattr(env_unwrapped, method_name)
            args = spec.get("args", spec)
            prepared_args = {}
            for args_name, args_data in args.items():
                value = args_data.get("value", None)
                is_in_dataset = args_data.get("in_dataset", True)
                if is_in_dataset:
                    if value not in callable_data:
                        print(
                            f"Col {value} not found in dataset, skipping callable "
                            f"for env {env_unwrapped}."
                        )
                        continue
                    prepared_args[args_name] = deepcopy(callable_data[value][env_idx])
                else:
                    prepared_args[args_name] = args_data.get("value")
            method(**prepared_args)

    state_restore_metrics = _collect_state_restore_metrics(world, current_step)
    if state_restore_metrics:
        print(
            "STATE_RESTORE_METRICS="
            f"{json.dumps(_make_json_safe(state_restore_metrics), sort_keys=True)}"
        )

    shape_prefix = world.infos["pixels"].shape[:2]
    if shape_prefix[1] != history_size:
        raise RuntimeError(
            "StableWM wrapper did not initialize the requested history size. "
            f"Expected {history_size}, got {shape_prefix[1]}."
        )
    goal_step = {
        key: np.broadcast_to(value[:, None, ...], shape_prefix + value.shape[1:])
        for key, value in goal_step_single.items()
    }

    world.infos.update(deepcopy(init_history))
    world.infos.update(deepcopy(goal_step))
    _set_stacked_history_buffers(world, init_history)

    if "goal" in goal_step and "goal" in world.infos:
        assert np.allclose(world.infos["goal"], goal_step["goal"]), (
            "Goal info does not match"
        )

    results = {
        "success_rate": 0.0,
        "episode_successes": np.zeros(len(episodes_idx)),
        "seeds": seeds,
    }
    results.update(state_restore_metrics)

    if target_frame_chunks:
        target_frames = np.stack(target_frame_chunks)
    else:
        target_frames = None

    video_frames = np.empty(
        (world.num_envs, eval_budget, *world.infos["pixels"].shape[-3:]),
        dtype=np.uint8,
    )
    frozen_infos = {}
    frozen_mask = np.zeros(world.num_envs, dtype=bool)

    for step_idx in range(eval_budget):
        video_frames[:, step_idx] = world.infos["pixels"][:, -1]
        world.infos.update(deepcopy(goal_step))
        world.step()
        current_successes = np.logical_or(
            results["episode_successes"], _successes_from_world(world)
        )
        newly_solved = np.logical_and(~frozen_mask, current_successes)
        if np.any(newly_solved):
            for key, value in world.infos.items():
                if isinstance(value, np.ndarray):
                    cache = frozen_infos.get(key)
                    if cache is None:
                        cache = np.empty_like(value)
                        frozen_infos[key] = cache
                    cache[newly_solved] = value[newly_solved]
            frozen_mask[newly_solved] = True

        if np.any(frozen_mask):
            for key, cache in frozen_infos.items():
                world.infos[key][frozen_mask] = cache[frozen_mask]

        results["episode_successes"] = current_successes
        world.envs.unwrapped._autoreset_envs = np.zeros((world.num_envs,))
        if np.all(results["episode_successes"]):
            if step_idx + 1 < eval_budget:
                last_frame = world.infos["pixels"][:, -1]
                remaining = eval_budget - (step_idx + 1)
                video_frames[:, step_idx + 1 :] = np.broadcast_to(
                    last_frame[:, None, ...],
                    (world.num_envs, remaining, *last_frame.shape[1:]),
                )
            break

    video_frames[:, -1] = world.infos["pixels"][:, -1]
    world.infos.update(deepcopy(goal_step))

    n_episodes = len(episodes_idx)
    results["success_rate"] = (
        float(np.sum(results["episode_successes"])) / n_episodes * 100.0
    )

    if save_video and target_frames is not None:
        import imageio

        target_len = target_frames.shape[1]
        video_path_obj = Path(video_path)
        video_path_obj.mkdir(parents=True, exist_ok=True)
        for env_idx in range(world.num_envs):
            out = imageio.get_writer(
                video_path_obj / f"rollout_{env_idx}.mp4",
                fps=15,
                codec="libx264",
            )
            goals = np.vstack([target_frames[env_idx, -1], target_frames[env_idx, -1]])
            for t in range(eval_budget):
                stacked_frame = np.vstack(
                    [video_frames[env_idx, t], target_frames[env_idx, t % target_len]]
                )
                frame = np.hstack([stacked_frame, goals])
                out.append_data(frame)
            out.close()
        print(f"Video saved to {video_path_obj}")

    if results["seeds"] is not None:
        assert np.unique(results["seeds"]).shape[0] == n_episodes, (
            "Some episode seeds are identical!"
        )

    _add_policy_debug_metrics(results, getattr(world, "policy", None), world)

    return results


def sample_fixed_offset_rows(dataset, ep_indices, cfg, eval_env_idx, goal_offset_steps):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = np.asarray(dataset.get_col_data(col_name)).reshape(-1)
    step_idx = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    valid_mask = step_idx <= np.array([max_start_idx_dict[ep] for ep in episode_idx])

    if eval_env_idx is not None and "env_idx" in dataset.column_names:
        valid_mask &= (
            np.asarray(dataset.get_col_data("env_idx")).reshape(-1) == eval_env_idx
        )
        print(
            f"Filtering dataset to env_idx={eval_env_idx} "
            f"({cfg.world.get('metaworld_env_name', 'unknown env')})."
        )

    valid_indices = np.nonzero(valid_mask)[0]
    print(len(valid_indices), "valid starting points found for evaluation.")
    if len(valid_indices) < cfg.eval.num_eval:
        raise ValueError(
            f"Not enough valid planning starts. Found {len(valid_indices)}, "
            f"need {cfg.eval.num_eval}."
        )

    g = np.random.default_rng(cfg.seed)
    selected_rows = np.sort(
        valid_indices[
            g.choice(len(valid_indices), size=cfg.eval.num_eval, replace=False)
        ]
    )
    rows = dataset.get_row_data(selected_rows)
    print(selected_rows)
    return rows[col_name], rows["step_idx"]


def sample_first_success_rows(dataset, ep_indices, cfg, eval_env_idx, success_key):
    if not dataset_has_key(dataset, success_key):
        raise KeyError(
            f"First-success goal sampling requires dataset key '{success_key}'. "
            "Regenerate and reconvert the eval dataset with the updated collector."
        )

    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = np.asarray(dataset.get_col_data(col_name)).reshape(-1)
    step_idx = np.asarray(dataset.get_col_data("step_idx")).reshape(-1)
    successes = success_mask(dataset, success_key)
    env_idx = None
    if eval_env_idx is not None and "env_idx" in dataset.column_names:
        env_idx = np.asarray(dataset.get_col_data("env_idx")).reshape(-1)
        print(
            f"Filtering dataset to env_idx={eval_env_idx} "
            f"({cfg.world.get('metaworld_env_name', 'unknown env')})."
        )

    start_offset = int(cfg.eval.goal_offset_steps)
    candidate_episodes = []
    candidate_starts = []

    for ep_id in ep_indices:
        mask = episode_idx == ep_id
        if env_idx is not None:
            mask &= env_idx == eval_env_idx
        if not np.any(mask):
            continue

        steps = step_idx[mask].astype(np.int64)
        ep_successes = successes[mask]
        order = np.argsort(steps)
        steps = steps[order]
        ep_successes = ep_successes[order]

        success_positions = np.flatnonzero(ep_successes)
        if success_positions.size == 0:
            continue

        goal_step = int(steps[success_positions[0]])
        start_step = goal_step - start_offset
        if start_step < int(steps[0]) or not np.any(steps == start_step):
            continue

        candidate_episodes.append(int(ep_id))
        candidate_starts.append(start_step)

    print(
        len(candidate_episodes),
        "valid first-success starting points found for evaluation.",
    )
    if len(candidate_episodes) < cfg.eval.num_eval:
        raise ValueError(
            "Not enough first-success planning starts. Found "
            f"{len(candidate_episodes)}, need {cfg.eval.num_eval}. "
            f"Collect more successful episodes or lower eval.goal_offset_steps."
        )

    g = np.random.default_rng(cfg.seed)
    selected = np.sort(
        g.choice(len(candidate_episodes), size=cfg.eval.num_eval, replace=False)
    )
    return (
        np.asarray(candidate_episodes, dtype=np.int64)[selected],
        np.asarray(candidate_starts, dtype=np.int64)[selected],
    )


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    keys_to_load = cfg.dataset.get("keys_to_load", None)
    if keys_to_load is not None:
        keys_to_load = list(keys_to_load)
        if str(cfg.eval.get("goal_sampling", "fixed_offset")) == "first_success":
            success_key = str(cfg.eval.get("goal_success_key", "success"))
            if success_key not in keys_to_load:
                keys_to_load.append(success_key)

    keys_to_cache = cfg.dataset.keys_to_cache
    if bool(cfg.dataset.get("cache_all_loaded", False)):
        keys_to_cache = keys_to_load

    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_load=keys_to_load,
        keys_to_cache=keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


@torch.no_grad()
def compute_final_latent_distance_metrics(policy, world):
    model = getattr(getattr(policy, "solver", None), "model", None)
    if model is None:
        return {}

    prepared_info, encoder, original_primary_source = policy.prepare_eval_info(world.infos)
    try:
        device = next(model.parameters()).device
        for key, value in list(prepared_info.items()):
            if torch.is_tensor(value):
                prepared_info[key] = value.to(device)

        model_encoder = getattr(model, "encoder", None)
        modality_encoders = getattr(model_encoder, "encoders", {})
        modality_sources = []
        image_sources = set()
        for _, modality_encoder in modality_encoders.items():
            source = getattr(modality_encoder, "source", None)
            if source is not None and source not in modality_sources:
                modality_sources.append(source)
            if source in {"pixels", "depth", "tactile"}:
                image_sources.add(source)

        def _ensure_sequence_value(source, value):
            if source in image_sources:
                expected_channels = {"pixels": 3, "depth": 1, "tactile": 1}.get(source)

                if value.ndim == 3:
                    return value.unsqueeze(1)

                if value.ndim == 4:
                    # Distinguish single-frame image tensors (B, C, H, W) or
                    # (B, H, W, C) from sequence tensors such as (B, T, H, W).
                    if expected_channels is not None:
                        if value.shape[1] == expected_channels or value.shape[-1] == expected_channels:
                            return value.unsqueeze(1)
                    return value

                return value
            if value.ndim == 2:
                return value.unsqueeze(1)
            return value

        current = {}
        for source in modality_sources:
            value = prepared_info.get(source)
            if torch.is_tensor(value):
                current[source] = _ensure_sequence_value(source, value)

        goal_source = getattr(model.encoder, "primary_source", "pixels")
        goal = {}
        for source in modality_sources:
            if source == goal_source:
                key = "goal"
            else:
                key = f"goal_{source}"
            value = prepared_info.get(key)
            if torch.is_tensor(value):
                goal[source] = _ensure_sequence_value(source, value)

        current = model.encode(current)
        goal = model.encode(goal)

        current_emb = current["emb"][:, -1]
        goal_emb = goal["emb"][:, -1]
        distances = F.mse_loss(current_emb, goal_emb.detach(), reduction="none").sum(dim=-1)
        distances_np = distances.detach().cpu().numpy()
        return {
            "final_latent_goal_distances": distances_np,
            "final_latent_goal_distance_mean": float(distances_np.mean()),
            "final_latent_goal_distance_variance": float(distances_np.var()),
        }
    finally:
        if encoder is not None and original_primary_source is not None:
            encoder.primary_source = original_primary_source


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    render_size = cfg.eval.get("render_size", cfg.eval.img_size)
    world = swm.World(**cfg.world, image_shape=(render_size, render_size))

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    keys_to_process = cfg.dataset.get("keys_to_process", cfg.dataset.keys_to_cache)
    for col in keys_to_process:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = np.asarray(stats_dataset.get_col_data(col))
        if col_data.ndim == 1:
            col_data = col_data[:, None]
        else:
            col_data = col_data.reshape(col_data.shape[0], -1)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")
    drop_modalities = get_drop_modalities(cfg.eval)
    modality_substitution = get_modality_substitution(cfg.eval)

    if policy != "random":
        model = swm.policy.AutoCostModel(cfg.policy)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        action_cost_cfg = cfg.eval.get("action_cost", {})
        action_norm_weight = float(action_cost_cfg.get("norm_weight", 0.0))
        action_delta_weight = float(action_cost_cfg.get("delta_weight", 0.0))
        first_action_delta_weight = float(
            action_cost_cfg.get("first_delta_weight", 0.0)
        )
        if (
            action_norm_weight != 0.0
            or action_delta_weight != 0.0
            or first_action_delta_weight != 0.0
        ):
            model = ActionRegularizedCostModel(
                model,
                action_processor=process.get("action"),
                action_norm_weight=action_norm_weight,
                action_delta_weight=action_delta_weight,
                first_action_delta_weight=first_action_delta_weight,
            )
            print(
                "Planning action regularization enabled: "
                f"norm={action_norm_weight}, "
                f"delta={action_delta_weight}, "
                f"first_delta={first_action_delta_weight}."
            )
        transform = build_eval_image_transform(cfg, model)
        if (
            drop_modalities
            and modality_substitution == "impute"
            and not model_supports_missing_modalities(model)
        ):
            raise ValueError(
                "eval.drop_modality / eval.drop_modalities requires a model whose "
                "fusion supports missing modalities. This checkpoint does not."
            )
        config = swm.PlanConfig(**cfg.plan_config)
        cfg.solver.device = device
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        if bool(cfg.eval.get("clamp_action_candidates", False)):
            solver = ActionClippedCEMSolver(
                solver,
                action_processor=process.get("action"),
                first_action_delta_limit=cfg.eval.get(
                    "first_action_delta_limit",
                    None,
                ),
                action_delta_limit=cfg.eval.get("action_delta_limit", None),
            )
            print("Clamping CEM candidates to the environment action bounds.")
            if cfg.eval.get("first_action_delta_limit", None) is not None:
                print(
                    "Limiting first action delta to "
                    f"{cfg.eval.first_action_delta_limit} in env action units."
                )
            if cfg.eval.get("action_delta_limit", None) is not None:
                print(
                    "Limiting all action deltas to "
                    f"{cfg.eval.action_delta_limit} in env action units."
                )
        policy = ModalityDropoutWorldModelPolicy(
            solver=solver,
            config=config,
            process=process,
            transform=transform,
            drop_modalities=drop_modalities,
            modality_substitution=modality_substitution,
            execution_action_delta_limit=cfg.eval.get(
                "execution_action_delta_limit",
                None,
            ),
            execution_action_norm_limit=cfg.eval.get(
                "execution_action_norm_limit",
                None,
            ),
        )
        if cfg.eval.get("execution_action_delta_limit", None) is not None:
            print(
                "Limiting executed action deltas to "
                f"{cfg.eval.execution_action_delta_limit} in env action units."
            )
        if cfg.eval.get("execution_action_norm_limit", None) is not None:
            print(
                "Limiting executed action norm to "
                f"{cfg.eval.execution_action_norm_limit} in env action units."
            )
        if drop_modalities:
            if modality_substitution == "zero":
                print(f"Zeroing modalities {drop_modalities} during evaluation.")
            else:
                print(f"Dropping modalities {drop_modalities} during evaluation.")

    else:
        policy = swm.policy.RandomPolicy()
        if drop_modalities:
            print(
                "Ignoring eval.drop_modality / eval.drop_modalities because the "
                "selected policy is random."
            )

    results_base_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )
    results_path = Path(cfg.output.filename)
    if not results_path.is_absolute():
        results_path = results_base_path / results_path

    video_path = (
        results_path.parent
        / "videos"
        / results_path.stem
        / f"seed_{cfg.seed}"
    )

    # For multi-task SensorMetaWorld datasets, keep only episodes for the env
    # instantiated by world.metaworld_env_name.
    eval_env_idx = resolve_dataset_env_idx(dataset, cfg)

    goal_sampling = str(cfg.eval.get("goal_sampling", "fixed_offset"))
    dataset_for_eval = dataset
    eval_goal_offset_steps = int(cfg.eval.goal_offset_steps)
    success_key = None

    if goal_sampling == "first_success":
        success_key = str(cfg.eval.get("goal_success_key", "success"))
        eval_goal_offset_steps = int(cfg.eval.goal_offset_steps) + 1
        dataset_for_eval = DatasetColumnFilter(dataset, hidden_keys={success_key})
        print(
            f"Using the first `{success_key}=True` step as the goal, with starts "
            f"{cfg.eval.goal_offset_steps} dataset steps before it."
        )

        eval_episodes, eval_start_idx = sample_first_success_rows(
            dataset=dataset,
            ep_indices=ep_indices,
            cfg=cfg,
            eval_env_idx=eval_env_idx,
            success_key=success_key,
        )
    else:
        eval_episodes, eval_start_idx = sample_fixed_offset_rows(
            dataset=dataset,
            ep_indices=ep_indices,
            cfg=cfg,
            eval_env_idx=eval_env_idx,
            goal_offset_steps=eval_goal_offset_steps,
        )

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    try:
        start_time = time.time()
        metrics = evaluate_from_dataset_with_history(
            world,
            dataset_for_eval,
            start_steps=eval_start_idx.tolist(),
            goal_offset_steps=eval_goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(
                cfg.eval.get("callables"), resolve=True
            ),
            save_video=bool(cfg.eval.get("save_video", True)),
            video_path=video_path,
        )
        if cfg.get("policy", "random") != "random":
            metrics.update(compute_final_latent_distance_metrics(policy, world))
        end_time = time.time()

        print(metrics)
        print("METRICS_JSON=" + json.dumps(_make_json_safe(metrics), sort_keys=True))

        results_path.parent.mkdir(parents=True, exist_ok=True)

        with results_path.open("a") as f:
            f.write("\n")  # separate from previous runs

            f.write("==== CONFIG ====\n")
            f.write(OmegaConf.to_yaml(cfg))
            f.write("\n")

            f.write("==== RESULTS ====\n")
            f.write(f"metrics: {metrics}\n")
            f.write(f"evaluation_time: {end_time - start_time} seconds\n")
    finally:
        world.close()


if __name__ == "__main__":
    run()
