"""Abstract interface that every dataset adapter must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod


class DatasetInterface(ABC):
    dataset_type = None

    @abstractmethod
    def build_dataset(
        self,
        cfg,
        cache_dir=None,
        extra_keys_to_load=None,
        passthrough_keys=None,
    ):
        """Build the runtime dataset and return (dataset, keys_to_load)."""

    @abstractmethod
    def populate_wm_dims(self, cfg, dataset, keys_to_load):
        """Populate modality dimensions on cfg.wm from the loaded dataset."""

    def add_conversion_args(self, parser):
        return parser

    def convert_from_args(self, args):
        raise NotImplementedError(
            f"Dataset type '{self.dataset_type}' does not implement conversion."
        )
