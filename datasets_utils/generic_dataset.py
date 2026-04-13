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
        from utils import (
            get_image_like_preprocessor,
            get_img_preprocessor,
        )

        transforms = []
        image_sources = set()
        passthrough_keys = set(passthrough_keys or [])
        vector_cfgs = {
            mod_cfg.get("source", name): mod_cfg
            for name, mod_cfg in get_vector_modality_configs(cfg.obs_encoder).items()
        }

        for _, mod_cfg in get_image_modality_configs(cfg.obs_encoder).items():
            source = mod_cfg.get("source")
            if source in image_sources:
                continue

            preprocess = mod_cfg.get(
                "preprocess",
                "imagenet" if source == "pixels" else "generic",
            )
            img_size = mod_cfg.get("img_size", cfg.img_size)
            if preprocess == "imagenet":
                transforms.append(
                    get_img_preprocessor(
                        source=source,
                        target=source,
                        img_size=img_size,
                    )
                )
            elif preprocess == "generic":
                transforms.append(
                    get_image_like_preprocessor(
                        source=source,
                        target=source,
                        img_size=img_size,
                        mean=mod_cfg.get("mean"),
                        std=mod_cfg.get("std"),
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported preprocess type '{preprocess}' for source '{source}'."
                )

            image_sources.add(source)

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
