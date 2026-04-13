"""Meta-World dataset adapter with saved-stat normalization and conversion hooks."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets_utils.generic_dataset import GenericDatasetAdapter
from datasets_utils.metaworld import converter
from datasets_utils.metaworld.preprocessor import get_saved_vector_normalizer


class MetaWorldDatasetAdapter(GenericDatasetAdapter):
    dataset_type = "metaworld"

    def get_vector_transform(self, dataset, col, mod_cfg=None):
        preprocess = mod_cfg.get("preprocess") if mod_cfg is not None else None
        if preprocess == "saved_stats":
            return get_saved_vector_normalizer(dataset, col, col)
        return super().get_vector_transform(dataset, col, mod_cfg=mod_cfg)

    def add_conversion_args(self, parser: argparse.ArgumentParser):
        parser.add_argument("src", type=Path, help="Input hierarchical HDF5 file.")
        parser.add_argument(
            "dst",
            type=Path,
            nargs="?",
            default=Path("~/.stable_worldmodel/metaworld.h5").expanduser(),
            help="Output flat HDF5 file. Defaults to ~/.stable_worldmodel/metaworld.h5",
        )
        parser.add_argument(
            "--keep-gripper-separate",
            action="store_true",
            help="Do not append the scalar gripper state onto the proprio vector.",
        )
        return parser

    def convert_from_args(self, args):
        converter.convert_dataset(
            src_path=args.src,
            dst_path=args.dst,
            merge_gripper=not args.keep_gripper_separate,
        )


DATASET_CLASS = MetaWorldDatasetAdapter
