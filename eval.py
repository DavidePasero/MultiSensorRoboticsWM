import os
import json
from collections import OrderedDict

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
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def model_supports_missing_modalities(model):
    encoder = getattr(model, "encoder", None)
    imputer = getattr(encoder, "imputer", None) if encoder is not None else None
    if imputer is not None:
        return bool(getattr(imputer, "supports_missing_modalities", False))
    fusion = getattr(encoder, "fusion", None) if encoder is not None else None
    return bool(getattr(fusion, "supports_missing_modalities", False))


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


class ModalityDropoutWorldModelPolicy(swm.policy.WorldModelPolicy):
    """World-model policy that removes a modality so missing-modality fusion can mask it."""

    def __init__(self, *args, drop_modalities=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_modalities = normalize_modalities_arg(drop_modalities)

    def _resolve_drop_sources(self):
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

    def get_action(self, info_dict: dict, **kwargs):
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        prepared_info = self._prepare_info(dict(info_dict))
        prepared_info, encoder, original_primary_source = (
            self._drop_selected_modalities(prepared_info)
        )

        try:
            if len(self._action_buffer) == 0:
                outputs = self.solver(prepared_info, init_action=self._next_init)

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

        return action

    def prepare_eval_info(self, info_dict: dict):
        prepared_info = self._prepare_info(dict(info_dict))
        prepared_info, encoder, original_primary_source = (
            self._drop_selected_modalities(prepared_info)
        )
        return prepared_info, encoder, original_primary_source


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


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

    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

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

    if policy != "random":
        model = swm.policy.AutoCostModel(cfg.policy)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        if drop_modalities and not model_supports_missing_modalities(model):
            raise ValueError(
                "eval.drop_modality / eval.drop_modalities requires a model whose "
                "fusion supports missing modalities. This checkpoint does not."
            )
        config = swm.PlanConfig(**cfg.plan_config)
        cfg.solver.device = device
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = ModalityDropoutWorldModelPolicy(
            solver=solver,
            config=config,
            process=process,
            transform=transform,
            drop_modalities=drop_modalities,
        )
        if drop_modalities:
            print(f"Dropping modalities {drop_modalities} during evaluation.")

    else:
        policy = swm.policy.RandomPolicy()
        if drop_modalities:
            print(
                "Ignoring eval.drop_modality / eval.drop_modalities because the "
                "selected policy is random."
            )

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
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
        metrics = world.evaluate_from_dataset(
            dataset_for_eval,
            start_steps=eval_start_idx.tolist(),
            goal_offset_steps=eval_goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(
                cfg.eval.get("callables"), resolve=True
            ),
            video_path=results_path,
        )
        if cfg.get("policy", "random") != "random":
            metrics.update(compute_final_latent_distance_metrics(policy, world))
        end_time = time.time()

        print(metrics)
        print("METRICS_JSON=" + json.dumps(_make_json_safe(metrics), sort_keys=True))

        results_path = results_path / cfg.output.filename
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
