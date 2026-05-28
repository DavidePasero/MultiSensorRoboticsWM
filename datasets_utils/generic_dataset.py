"""Default adapter for flat HDF5 datasets that use the repository's generic transforms."""

from __future__ import annotations

from datasets_utils.dataset_interface import DatasetInterface


class GenericDatasetAdapter(DatasetInterface):
    dataset_type = "generic"

    def resolve_keys_to_load(self, cfg, extra_keys_to_load=None):
        keys_to_load = list(cfg.data.dataset.keys_to_load)
        for key in extra_keys_to_load or []:
            if key not in keys_to_load:
                keys_to_load.append(key)
        return keys_to_load

    def build_hdf5_dataset(self, cfg, keys_to_load, cache_dir=None):
        import stable_worldmodel as swm
        from datasets_utils.sharded_hdf5 import ShardedHDF5Dataset

        shard_glob = cfg.data.dataset.get("shard_glob")
        shard_paths = cfg.data.dataset.get("shard_paths")
        if shard_glob or shard_paths:
            return ShardedHDF5Dataset(
                shard_glob=shard_glob,
                shard_paths=list(shard_paths or []),
                frameskip=int(cfg.data.dataset.frameskip),
                num_steps=int(cfg.data.dataset.num_steps),
                keys_to_load=keys_to_load,
                keys_to_cache=list(cfg.data.dataset.get("keys_to_cache", [])),
                keys_to_merge=dict(cfg.data.dataset.get("keys_to_merge", {})),
                cache_dir=cache_dir or swm.data.utils.get_cache_dir(),
                transform=None,
                max_cached_shards_per_worker=int(
                    cfg.data.dataset.get("max_cached_shards_per_worker", 1)
                ),
            )

        return swm.data.HDF5Dataset(
            name=cfg.data.dataset.name,
            frameskip=int(cfg.data.dataset.frameskip),
            num_steps=int(cfg.data.dataset.num_steps),
            keys_to_load=keys_to_load,
            keys_to_cache=list(cfg.data.dataset.get("keys_to_cache", [])),
            keys_to_merge=dict(cfg.data.dataset.get("keys_to_merge", {})),
            cache_dir=cache_dir or swm.data.utils.get_cache_dir(),
            transform=None,
        )

    def build_dataset(
        self,
        cfg,
        cache_dir=None,
        extra_keys_to_load=None,
        passthrough_keys=None,
    ):
        keys_to_load = self.resolve_keys_to_load(cfg, extra_keys_to_load)
        dataset = self.build_hdf5_dataset(cfg, keys_to_load, cache_dir=cache_dir)
        dataset.transform = self.build_transform(
            cfg,
            dataset,
            keys_to_load=keys_to_load,
            passthrough_keys=passthrough_keys,
        )
        return dataset, keys_to_load

    def build_transform(
        self,
        cfg,
        dataset,
        *,
        keys_to_load,
        passthrough_keys=None,
    ):
        import stable_pretraining as spt

        from multimodal import get_image_modality_configs, get_vector_modality_configs

        transforms = []
        image_sources = {
            mod_cfg.get("source", name)
            for name, mod_cfg in get_image_modality_configs(cfg.obs_encoder).items()
        }
        passthrough_keys = set(passthrough_keys or [])
        vector_cfgs = {
            mod_cfg.get("source", name): mod_cfg
            for name, mod_cfg in get_vector_modality_configs(cfg.obs_encoder).items()
        }

        # Image modalities stay raw in the dataset so preprocessing can run on the model device.

        for col in keys_to_load:
            if col in image_sources or col in passthrough_keys:
                continue

            mod_cfg = vector_cfgs.get(col)
            transforms.append(self.get_vector_transform(dataset, col, mod_cfg))

        return spt.data.transforms.Compose(*transforms) if transforms else None

    def get_vector_transform(self, dataset, col, mod_cfg=None):
        from utils import get_column_normalizer

        preprocess = mod_cfg.get("preprocess") if mod_cfg is not None else None
        if preprocess in (None, "zscore"):
            return get_column_normalizer(dataset, col, col)

        raise ValueError(
            f"Unsupported vector preprocess '{preprocess}' for column '{col}' "
            f"on dataset type '{self.dataset_type}'."
        )

    def populate_wm_dims(self, cfg, dataset, keys_to_load):
        from omegaconf import open_dict

        from multimodal import get_image_modality_configs

        image_sources = {
            mod_cfg.get("source")
            for _, mod_cfg in get_image_modality_configs(cfg.obs_encoder).items()
        }

        with open_dict(cfg):
            for col in keys_to_load:
                if col in image_sources:
                    continue
                setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))


DATASET_CLASS = GenericDatasetAdapter
