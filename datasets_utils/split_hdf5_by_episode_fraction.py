"""Split a flat HDF5 dataset into two complete-episode subsets.

Example:
    python datasets_utils/split_hdf5_by_episode_fraction.py \
      metaworld_bin_picking.h5 \
      metaworld_bin_picking_70.h5 \
      metaworld_bin_picking_30.h5 \
      --fraction 0.7 --shuffle --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


ROW_METADATA_KEYS = {"ep_len", "ep_offset"}
STATS_GROUP_NAME = "stats"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split a converted flat HDF5 dataset into two files by percentage "
            "of complete episodes. The first output receives --fraction of the "
            "episodes; the second receives the rest."
        )
    )
    parser.add_argument("src", type=Path, help="Input flat .h5 dataset.")
    parser.add_argument("dst_a", type=Path, help="Output .h5 for the first split.")
    parser.add_argument("dst_b", type=Path, help="Output .h5 for the second split.")
    parser.add_argument(
        "--fraction",
        type=float,
        required=True,
        help="Fraction of episodes to place in the first split, e.g. 0.7.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle episodes before splitting. The output episode order is sorted back to source order.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--preserve-episode-idx",
        action="store_true",
        help="Copy original episode_idx values instead of renumbering within each split.",
    )
    return parser.parse_args()


def validate_source(src_file):
    for key in ("ep_len", "ep_offset"):
        if key not in src_file:
            raise KeyError(f"Input dataset is missing required key '{key}'.")

    ep_len = np.asarray(src_file["ep_len"][:], dtype=np.int64)
    ep_offset = np.asarray(src_file["ep_offset"][:], dtype=np.int64)
    if ep_len.ndim != 1 or ep_offset.ndim != 1:
        raise ValueError("ep_len and ep_offset must be 1D arrays.")
    if len(ep_len) != len(ep_offset):
        raise ValueError("ep_len and ep_offset must have the same number of episodes.")
    if len(ep_len) < 2:
        raise ValueError("Need at least two episodes to split a dataset.")
    if np.any(ep_len <= 0):
        raise ValueError("All episodes must have positive length.")
    return ep_len, ep_offset


def split_episode_indices(num_episodes: int, fraction: float, *, shuffle: bool, seed: int):
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"--fraction must be in (0, 1), got {fraction}.")

    indices = np.arange(num_episodes, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    split_at = int(round(num_episodes * fraction))
    split_at = min(max(split_at, 1), num_episodes - 1)
    first = np.sort(indices[:split_at])
    second = np.sort(indices[split_at:])
    return first, second


def copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def create_like_dataset(dst_file, src_dataset, name: str, total_rows: int):
    kwargs = {}
    if src_dataset.chunks is not None:
        chunk_rows = min(src_dataset.chunks[0], max(total_rows, 1))
        kwargs["chunks"] = (chunk_rows, *src_dataset.chunks[1:])
    if src_dataset.compression is not None:
        kwargs["compression"] = src_dataset.compression
        kwargs["compression_opts"] = src_dataset.compression_opts
        kwargs["shuffle"] = src_dataset.shuffle
        kwargs["fletcher32"] = src_dataset.fletcher32

    dst_dataset = dst_file.create_dataset(
        name,
        shape=(total_rows, *src_dataset.shape[1:]),
        dtype=src_dataset.dtype,
        **kwargs,
    )
    copy_attrs(src_dataset, dst_dataset)
    return dst_dataset


def copy_non_row_items(src_file, dst_file, row_count: int):
    for key, item in src_file.items():
        if key in ROW_METADATA_KEYS or key == STATS_GROUP_NAME:
            continue
        if isinstance(item, h5py.Dataset) and item.shape[:1] == (row_count,):
            continue
        src_file.copy(key, dst_file)


def recompute_force_torque_stats(dst_file):
    if "force_torque" not in dst_file:
        return
    values = np.asarray(dst_file["force_torque"][:], dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        return
    mean = values.mean(axis=0).astype(np.float32)
    var = np.maximum(values.var(axis=0), 1e-12)
    std = np.sqrt(var).astype(np.float32)

    stats = dst_file.require_group(STATS_GROUP_NAME)
    for name, data in {
        "force_torque_mean": mean,
        "force_torque_std": std,
    }.items():
        if name in stats:
            del stats[name]
        stats.create_dataset(name, data=data)


def write_split(
    src_file,
    dst_path: Path,
    episode_indices,
    ep_len,
    ep_offset,
    *,
    preserve_episode_idx: bool,
    split_name: str,
):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {dst_path}")

    row_count = int(ep_len.sum())
    selected_lengths = ep_len[episode_indices].astype(np.int64)
    total_rows = int(selected_lengths.sum())
    selected_offsets = np.zeros_like(selected_lengths)
    if len(selected_offsets) > 1:
        selected_offsets[1:] = np.cumsum(selected_lengths[:-1], dtype=np.int64)

    row_dataset_keys = [
        key
        for key, item in src_file.items()
        if isinstance(item, h5py.Dataset)
        and key not in ROW_METADATA_KEYS
        and item.shape[:1] == (row_count,)
    ]

    with h5py.File(dst_path, "w") as dst_file:
        copy_attrs(src_file, dst_file)
        dst_file.attrs["source_dataset"] = str(src_file.filename)
        dst_file.attrs["split_name"] = split_name
        dst_file.attrs["source_episode_indices_json"] = json.dumps(
            [int(idx) for idx in episode_indices]
        )

        ep_len_ds = dst_file.create_dataset("ep_len", data=selected_lengths, dtype=np.int64)
        ep_offset_ds = dst_file.create_dataset(
            "ep_offset", data=selected_offsets, dtype=np.int64
        )
        if "ep_len" in src_file:
            copy_attrs(src_file["ep_len"], ep_len_ds)
        if "ep_offset" in src_file:
            copy_attrs(src_file["ep_offset"], ep_offset_ds)

        dst_datasets = {
            key: create_like_dataset(dst_file, src_file[key], key, total_rows)
            for key in row_dataset_keys
        }

        write_offset = 0
        for new_episode_idx, src_episode_idx in enumerate(episode_indices):
            src_episode_idx = int(src_episode_idx)
            length = int(ep_len[src_episode_idx])
            src_start = int(ep_offset[src_episode_idx])
            src_slice = slice(src_start, src_start + length)
            dst_slice = slice(write_offset, write_offset + length)

            for key, dst_dataset in dst_datasets.items():
                if key == "episode_idx" and not preserve_episode_idx:
                    dst_dataset[dst_slice] = new_episode_idx
                elif key == "step_idx":
                    dst_dataset[dst_slice] = np.arange(length, dtype=dst_dataset.dtype)
                else:
                    dst_dataset[dst_slice] = src_file[key][src_slice]

            write_offset += length

        copy_non_row_items(src_file, dst_file, row_count)
        recompute_force_torque_stats(dst_file)

    return {
        "path": str(dst_path),
        "episodes": int(len(episode_indices)),
        "total_steps": total_rows,
        "mean_ep_len": float(selected_lengths.mean()),
        "min_ep_len": int(selected_lengths.min()),
        "max_ep_len": int(selected_lengths.max()),
    }


def main():
    args = parse_args()
    with h5py.File(args.src, "r") as src_file:
        ep_len, ep_offset = validate_source(src_file)
        first, second = split_episode_indices(
            len(ep_len),
            args.fraction,
            shuffle=args.shuffle,
            seed=args.seed,
        )
        summary = {
            "source": str(args.src),
            "source_episodes": int(len(ep_len)),
            "source_total_steps": int(ep_len.sum()),
            "fraction": float(args.fraction),
            "shuffle": bool(args.shuffle),
            "seed": int(args.seed),
            "first": write_split(
                src_file,
                args.dst_a,
                first,
                ep_len,
                ep_offset,
                preserve_episode_idx=args.preserve_episode_idx,
                split_name="first",
            ),
            "second": write_split(
                src_file,
                args.dst_b,
                second,
                ep_len,
                ep_offset,
                preserve_episode_idx=args.preserve_episode_idx,
                split_name="second",
            ),
        }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
