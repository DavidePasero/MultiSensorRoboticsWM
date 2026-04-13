"""Registry and factory helpers for loading dataset adapters by dataset type."""

from __future__ import annotations

from importlib import import_module
from functools import lru_cache


DATASET_MODULES = {
    "generic": "datasets_utils.generic_dataset",
    "metaworld": "datasets_utils.metaworld.dataset",
}


def get_registered_dataset_types():
    return sorted(DATASET_MODULES)


@lru_cache(maxsize=None)
def _load_dataset_module(dataset_type: str):
    if dataset_type not in DATASET_MODULES:
        known = ", ".join(get_registered_dataset_types())
        raise ValueError(
            f"Unknown dataset type '{dataset_type}'. Known dataset types: {known}."
        )

    module_name = DATASET_MODULES[dataset_type]
    return import_module(module_name)


def get_dataset_adapter(dataset_type: str):
    module = _load_dataset_module(dataset_type)
    dataset_cls = getattr(module, "DATASET_CLASS", None)
    if dataset_cls is None:
        raise AttributeError(
            f"Dataset module for '{dataset_type}' does not define DATASET_CLASS."
        )
    return dataset_cls()


def get_dataset_adapter_from_config(cfg):
    dataset_type = cfg.data.dataset.get("type", "generic")
    return get_dataset_adapter(str(dataset_type))
