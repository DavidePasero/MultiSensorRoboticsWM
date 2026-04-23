import os
import json
from collections import OrderedDict

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import h5py
import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


def model_supports_missing_modalities(model):
    encoder = getattr(model, "encoder", None)
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


class ModalityDropoutWorldModelPolicy(swm.policy.WorldModelPolicy):
    """World-model policy that removes a modality so missing-modality fusion can mask it."""

    def __init__(self, *args, drop_modality=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.drop_modality = None if drop_modality in {None, "", "none"} else str(drop_modality)

    def _resolve_drop_source(self):
        if self.drop_modality is None:
            return None

        model = getattr(self.solver, "model", None)
        modality_sources = get_model_modality_sources(model)
        if not modality_sources:
            return self.drop_modality

        if self.drop_modality in modality_sources:
            return modality_sources[self.drop_modality]

        if self.drop_modality in modality_sources.values():
            return self.drop_modality

        known = sorted(set(modality_sources) | set(modality_sources.values()))
        raise ValueError(
            f"Unknown modality '{self.drop_modality}'. Available modalities: {known}."
        )

    def _find_fallback_primary_source(self, info_dict, dropped_source):
        model = getattr(self.solver, "model", None)
        modality_sources = get_model_modality_sources(model)
        for source in modality_sources.values():
            if source == dropped_source or source not in info_dict:
                continue

            goal_key = "goal" if source == "pixels" else f"goal_{source}"
            if goal_key in info_dict:
                return source
        return None

    def _drop_selected_modality(self, info_dict):
        drop_source = self._resolve_drop_source()
        if drop_source is None:
            return info_dict, None, None

        pruned = dict(info_dict)
        pruned.pop(drop_source, None)
        pruned.pop("goal" if drop_source == "pixels" else f"goal_{drop_source}", None)

        model = getattr(self.solver, "model", None)
        encoder = getattr(model, "encoder", None)
        original_primary_source = getattr(encoder, "primary_source", None)

        if encoder is not None and original_primary_source == drop_source:
            fallback_source = self._find_fallback_primary_source(pruned, drop_source)
            if fallback_source is None:
                raise ValueError(
                    f"Cannot drop primary modality '{drop_source}' because no fallback "
                    "goal-conditioned modality is available."
                )

            fallback_goal_key = "goal" if fallback_source == "pixels" else f"goal_{fallback_source}"
            encoder.primary_source = fallback_source
            pruned["goal"] = pruned[fallback_goal_key]

        return pruned, encoder, original_primary_source

    def get_action(self, info_dict: dict, **kwargs):
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        prepared_info = self._prepare_info(dict(info_dict))
        prepared_info, encoder, original_primary_source = self._drop_selected_modality(
            prepared_info
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


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_load=cfg.dataset.get("keys_to_load", None),
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


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

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")
    drop_modality = cfg.eval.get("drop_modality", None)

    if policy != "random":
        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        if drop_modality is not None and not model_supports_missing_modalities(model):
            raise ValueError(
                "eval.drop_modality requires a model whose fusion supports missing "
                "modalities. This checkpoint does not."
            )
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = ModalityDropoutWorldModelPolicy(
            solver=solver,
            config=config,
            process=process,
            transform=transform,
            drop_modality=drop_modality,
        )
        if drop_modality is not None:
            print(f"Dropping modality '{drop_modality}' during evaluation.")

    else:
        policy = swm.policy.RandomPolicy()
        if drop_modality is not None:
            print("Ignoring eval.drop_modality because the selected policy is random.")

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove rows that are too close to the end of the episode
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row

    # For multi-task SensorMetaWorld datasets, keep only episodes for the env
    # instantiated by world.metaworld_env_name.
    eval_env_idx = resolve_dataset_env_idx(dataset, cfg)
    if eval_env_idx is not None:
        valid_mask = valid_mask & (dataset.get_col_data("env_idx") == eval_env_idx)
        print(
            f"Filtering dataset to env_idx={eval_env_idx} "
            f"({cfg.world.get('metaworld_env_name', 'unknown env')})."
        )

    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=results_path,
    )
    end_time = time.time()
    
    print(metrics)

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


if __name__ == "__main__":
    run()
