"""Sharded HDF5 dataset utilities for reducing monolithic-file contention."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
import glob
import re

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from torch.utils.data import Sampler, Subset

from stable_worldmodel.data.dataset import Dataset
from stable_worldmodel.data.utils import get_cache_dir

from datasets_utils.metaworld.preprocessor import load_vector_stats


def _resolve_shard_paths(
    *,
    shard_glob: str | None = None,
    shard_paths: list[str] | None = None,
    cache_dir: str | Path | None = None,
) -> list[Path]:
    base_dir = Path(cache_dir or get_cache_dir())
    paths: list[Path] = []

    if shard_glob:
        pattern = str(Path(shard_glob).expanduser())
        if not Path(pattern).is_absolute():
            pattern = str(base_dir / shard_glob)
        paths.extend(Path(p) for p in glob.glob(pattern))

    for shard_path in shard_paths or []:
        path = Path(shard_path).expanduser()
        if not path.is_absolute():
            path = base_dir / shard_path
        paths.append(path)

    unique_paths = sorted({path.resolve() for path in paths})
    if not unique_paths:
        raise ValueError(
            "No shard files were found. Set data.dataset.shard_glob or "
            "data.dataset.shard_paths to point at converted flat .h5 shards."
        )
    return unique_paths


class ShardedHDF5Dataset(Dataset):
    """Map-style dataset over many flat HDF5 shards with lazy per-worker shard caches."""

    is_sharded = True

    def __init__(
        self,
        *,
        shard_glob: str | None = None,
        shard_paths: list[str] | None = None,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        keys_to_merge: dict[str, list[str] | str] | None = None,
        cache_dir: str | Path | None = None,
        max_cached_shards_per_worker: int = 1,
    ) -> None:
        self.shard_paths = _resolve_shard_paths(
            shard_glob=shard_glob,
            shard_paths=shard_paths,
            cache_dir=cache_dir,
        )
        self.cache_dir = Path(cache_dir or get_cache_dir())
        self.h5_path = self.shard_paths[0]
        self.max_cached_shards_per_worker = max(int(max_cached_shards_per_worker), 1)
        self.keys_to_cache = list(keys_to_cache or [])
        self._keys: list[str] | None = keys_to_load
        self._active_shards: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._merged_cache: dict[str, np.ndarray] = {}
        self._row_ranges: list[tuple[int, int]] = []
        self._shards: list[dict[str, Any]] = []

        global_lengths = []
        global_row_offset = 0
        clip_shard_indices = []
        clip_episode_indices = []
        clip_start_indices = []
        shard_clip_indices = defaultdict(list)

        for shard_idx, shard_path in enumerate(self.shard_paths):
            with h5py.File(shard_path, "r") as h5_file:
                lengths = np.asarray(h5_file["ep_len"][:], dtype=np.int64)
                offsets = np.asarray(h5_file["ep_offset"][:], dtype=np.int64)
                shard_keys = keys_to_load or [
                    key for key in h5_file.keys() if key not in ("ep_len", "ep_offset")
                ]
                if self._keys is None:
                    self._keys = list(shard_keys)
                elif list(shard_keys) != list(self._keys):
                    missing = sorted(set(self._keys) - set(shard_keys))
                    extra = sorted(set(shard_keys) - set(self._keys))
                    raise ValueError(
                        f"Shard {shard_path} does not match expected keys. "
                        f"Missing={missing}, extra={extra}"
                    )

                shard_rows = int(lengths.sum())
                self._shards.append(
                    {
                        "path": shard_path,
                        "lengths": lengths,
                        "offsets": offsets,
                        "rows": shard_rows,
                    }
                )
                self._row_ranges.append((global_row_offset, global_row_offset + shard_rows))
                global_row_offset += shard_rows
                global_lengths.extend(lengths.tolist())

                for ep_idx, length in enumerate(lengths):
                    if length < num_steps * frameskip:
                        continue
                    for start in range(length - num_steps * frameskip + 1):
                        sample_idx = len(clip_start_indices)
                        clip_shard_indices.append(shard_idx)
                        clip_episode_indices.append(ep_idx)
                        clip_start_indices.append(start)
                        shard_clip_indices[shard_idx].append(sample_idx)

        global_lengths_arr = np.asarray(global_lengths, dtype=np.int64)
        global_offsets = np.zeros_like(global_lengths_arr)
        if len(global_offsets) > 1:
            global_offsets[1:] = np.cumsum(global_lengths_arr[:-1], dtype=np.int64)

        # Avoid Dataset.__init__ here: it eagerly materializes a huge Python
        # clip_indices list, which we replace anyway and which gets replicated
        # poorly across DataLoader worker forks.
        self.lengths = global_lengths_arr
        self.offsets = global_offsets
        self.frameskip = int(frameskip)
        self.num_steps = int(num_steps)
        self.span = self.num_steps * self.frameskip
        self.transform = transform
        self.clip_shard_indices = np.asarray(clip_shard_indices, dtype=np.int32)
        self.clip_episode_indices = np.asarray(clip_episode_indices, dtype=np.int32)
        self.clip_start_indices = np.asarray(clip_start_indices, dtype=np.int32)
        self.shard_clip_indices = {
            shard_idx: np.asarray(indices, dtype=np.int64)
            for shard_idx, indices in shard_clip_indices.items()
        }

        if keys_to_merge:
            for target, source in keys_to_merge.items():
                self.merge_col(source, target)

    @property
    def column_names(self) -> list[str]:
        return list(self._keys)

    def __len__(self) -> int:
        return int(self.clip_start_indices.shape[0])

    def get_clip_location(self, idx: int) -> tuple[int, int, int]:
        return (
            int(self.clip_shard_indices[idx]),
            int(self.clip_episode_indices[idx]),
            int(self.clip_start_indices[idx]),
        )

    def _evict_extra_shards(self) -> None:
        while len(self._active_shards) > self.max_cached_shards_per_worker:
            _oldest_idx, state = self._active_shards.popitem(last=False)
            h5_file = state.get("h5_file")
            if h5_file is not None:
                h5_file.close()

    def _ensure_shard_state(self, shard_idx: int) -> dict[str, Any]:
        state = self._active_shards.get(shard_idx)
        if state is not None:
            self._active_shards.move_to_end(shard_idx)
            return state

        shard_path = self._shards[shard_idx]["path"]
        h5_file = h5py.File(
            shard_path,
            "r",
            swmr=True,
            rdcc_nbytes=256 * 1024 * 1024,
        )
        cache = {}
        for key in self.keys_to_cache:
            cache[key] = h5_file[key][:]
        state = {"h5_file": h5_file, "cache": cache}
        self._active_shards[shard_idx] = state
        self._evict_extra_shards()
        return state

    def _load_slice_from_shard(
        self,
        shard_idx: int,
        ep_idx: int,
        start: int,
        end: int,
    ) -> dict[str, Any]:
        state = self._ensure_shard_state(shard_idx)
        shard = self._shards[shard_idx]
        g_start = int(shard["offsets"][ep_idx] + start)
        g_end = int(shard["offsets"][ep_idx] + end)
        steps = {}
        for col in self._keys:
            src = state["cache"] if col in state["cache"] else state["h5_file"]
            data = src[col][g_start:g_end]
            if col != "action":
                data = data[:: self.frameskip]

            if data.dtype == np.object_ or data.dtype.kind in ("S", "U"):
                val = data[0] if len(data) > 0 else b""
                steps[col] = val.decode() if isinstance(val, bytes) else val
            else:
                tensor = torch.from_numpy(data)
                if tensor.ndim == 4 and tensor.shape[-1] in (1, 3):
                    tensor = tensor.permute(0, 3, 1, 2)
                steps[col] = tensor
        return self.transform(steps) if self.transform else steps

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard_idx, ep_idx, start = self.get_clip_location(idx)
        steps = self._load_slice_from_shard(
            shard_idx,
            ep_idx,
            start,
            start + self.span,
        )
        if "action" in steps:
            steps["action"] = steps["action"].reshape(self.num_steps, -1)
        return steps

    def load_chunk(
        self, episodes_idx: np.ndarray, start: np.ndarray, end: np.ndarray
    ) -> list[dict]:
        chunk = []
        for sample_idx, s, e in zip(episodes_idx, start, end):
            shard_idx, ep_idx, _ = self.get_clip_location(int(sample_idx))
            steps = self._load_slice_from_shard(shard_idx, ep_idx, int(s), int(e))
            if "action" in steps:
                steps["action"] = steps["action"].reshape(
                    (int(e) - int(s)) // self.frameskip,
                    -1,
                )
            chunk.append(steps)
        return chunk

    def load_episode(self, episode_idx: int) -> dict:
        raise NotImplementedError(
            "Episode-indexed loading is not implemented for sharded datasets."
        )

    def _get_col(self, col: str) -> np.ndarray:
        if col in self._merged_cache:
            return self._merged_cache[col]
        arrays = []
        for shard in self._shards:
            with h5py.File(shard["path"], "r") as h5_file:
                arrays.append(h5_file[col][:])
        return np.concatenate(arrays, axis=0)

    def get_col_data(self, col: str) -> np.ndarray:
        return self._get_col(col)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        if isinstance(row_idx, list):
            return {col: self.get_col_data(col)[row_idx] for col in self._keys}
        return {col: self.get_col_data(col)[row_idx] for col in self._keys}

    def merge_col(
        self,
        source: list[str] | str,
        target: str,
        dim: int = -1,
    ) -> None:
        if isinstance(source, str):
            source = [key for key in self._keys if re.match(source, key)]
        merged = np.concatenate([self._get_col(src_key) for src_key in source], axis=dim)
        self._merged_cache[target] = merged
        if target not in self._keys:
            self._keys.append(target)

    def get_dim(self, col: str) -> int:
        data = self.get_col_data(col)
        return np.prod(data.shape[1:]).item() if data.ndim > 1 else 1

    def load_saved_vector_stats(self, column: str):
        counts = []
        means = []
        second_moments = []
        for shard in self._shards:
            count = int(shard["lengths"].sum())
            mean_t, std_t = load_vector_stats(shard["path"], column)
            mean = mean_t.numpy()
            std = std_t.numpy()
            counts.append(count)
            means.append(mean)
            second_moments.append(np.square(std) + np.square(mean))

        total_count = int(sum(counts))
        if total_count <= 0:
            raise ValueError(f"No rows available to aggregate stats for '{column}'.")

        mean = sum(count * mean for count, mean in zip(counts, means)) / total_count
        second = (
            sum(count * second for count, second in zip(counts, second_moments))
            / total_count
        )
        var = np.maximum(second - np.square(mean), 1e-12)
        std = np.sqrt(var)
        return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(
            std.astype(np.float32)
        )


def unwrap_subset(dataset):
    if isinstance(dataset, Subset):
        return dataset.dataset, list(dataset.indices)
    return dataset, list(range(len(dataset)))


def uses_shard_local_batches(dataset) -> bool:
    base_dataset, _ = unwrap_subset(dataset)
    return bool(getattr(base_dataset, "is_sharded", False))


class ShardLocalBatchSampler(Sampler[list[int]]):
    """Build batches that stay within one shard to maximize per-worker locality."""

    def __init__(
        self,
        dataset,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self._epoch = 0

        base_dataset, outer_indices = unwrap_subset(dataset)
        if not getattr(base_dataset, "is_sharded", False):
            raise ValueError("ShardLocalBatchSampler requires a ShardedHDF5Dataset.")

        shard_to_positions = defaultdict(list)
        for outer_pos, base_idx in enumerate(outer_indices):
            shard_idx = base_dataset.get_clip_location(int(base_idx))[0]
            shard_to_positions[shard_idx].append(outer_pos)
        self._shard_to_positions = {
            shard_idx: np.asarray(positions, dtype=np.int64)
            for shard_idx, positions in shard_to_positions.items()
        }
        self._num_batches = None

    def __len__(self) -> int:
        if self._num_batches is None:
            total = 0
            for positions in self._shard_to_positions.values():
                if self.drop_last:
                    total += len(positions) // self.batch_size
                else:
                    total += (len(positions) + self.batch_size - 1) // self.batch_size
            self._num_batches = total
        return self._num_batches

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch)
        self._epoch += 1

        shard_order = list(self._shard_to_positions.keys())
        if self.shuffle:
            perm = torch.randperm(len(shard_order), generator=generator).tolist()
            shard_order = [shard_order[idx] for idx in perm]

        for shard_idx in shard_order:
            positions = self._shard_to_positions[shard_idx]
            if self.shuffle:
                shard_perm = torch.randperm(len(positions), generator=generator).tolist()
                positions = positions[shard_perm]

            for start in range(0, len(positions), self.batch_size):
                batch = positions[start : start + self.batch_size].tolist()
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                yield batch
