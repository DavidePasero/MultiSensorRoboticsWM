import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def iter_episodes(src_file):
    for env_name in sorted(src_file.keys()):
        env_group = src_file[env_name]
        for episode_name in sorted(env_group.keys()):
            yield env_name, episode_name, env_group[episode_name]


def episode_length(episode_group):
    return int(episode_group["action"].shape[0])


def infer_shapes(src_file, merge_gripper):
    for _env_name, _episode_name, episode_group in iter_episodes(src_file):
        shapes = {
            "pixels": episode_group["pixels"].shape[1:],
            "depth": episode_group["depth"].shape[1:],
            "tactile": episode_group["tactile"].shape[1:],
            "action": episode_group["action"].shape[1:],
        }

        proprio = episode_group["proprio"][()]
        if merge_gripper and "gripper" in episode_group:
            gripper = episode_group["gripper"][()]
            shapes["proprio"] = (proprio.shape[-1] + 1,)
            shapes["gripper"] = gripper.shape[1:] if gripper.ndim > 1 else ()
        else:
            shapes["proprio"] = proprio.shape[1:]
        return shapes

    raise ValueError("No episodes found in source Meta-World dataset.")


def create_dataset(out_file, name, total_steps, feature_shape, dtype):
    return out_file.create_dataset(
        name,
        shape=(total_steps, *feature_shape),
        dtype=dtype,
    )


def convert_dataset(src_path, dst_path, merge_gripper=True):
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, "r") as src_file:
        episodes = list(iter_episodes(src_file))
        if not episodes:
            raise ValueError(f"No episodes found in {src_path}.")

        total_steps = sum(episode_length(ep_group) for _, _, ep_group in episodes)
        shapes = infer_shapes(src_file, merge_gripper=merge_gripper)
        episode_lengths = np.array(
            [episode_length(ep_group) for _, _, ep_group in episodes],
            dtype=np.int64,
        )
        episode_offsets = np.zeros_like(episode_lengths)
        if len(episode_offsets) > 1:
            episode_offsets[1:] = np.cumsum(episode_lengths[:-1], dtype=np.int64)

        env_names = sorted({env_name for env_name, _, _ in episodes})
        env_to_idx = {env_name: idx for idx, env_name in enumerate(env_names)}

        with h5py.File(dst_path, "w") as dst_file:
            pixels_ds = create_dataset(dst_file, "pixels", total_steps, shapes["pixels"], np.uint8)
            depth_ds = create_dataset(dst_file, "depth", total_steps, shapes["depth"], np.float32)
            tactile_ds = create_dataset(dst_file, "tactile", total_steps, shapes["tactile"], np.float32)
            proprio_ds = create_dataset(dst_file, "proprio", total_steps, shapes["proprio"], np.float32)
            action_ds = create_dataset(dst_file, "action", total_steps, shapes["action"], np.float32)
            ep_len_ds = dst_file.create_dataset("ep_len", data=episode_lengths, dtype=np.int64)
            ep_offset_ds = dst_file.create_dataset("ep_offset", data=episode_offsets, dtype=np.int64)
            episode_idx_ds = dst_file.create_dataset("episode_idx", shape=(total_steps,), dtype=np.int64)
            step_idx_ds = dst_file.create_dataset("step_idx", shape=(total_steps,), dtype=np.int64)
            env_idx_ds = dst_file.create_dataset("env_idx", shape=(total_steps,), dtype=np.int64)

            offset = 0
            for episode_idx, (env_name, episode_name, episode_group) in enumerate(episodes):
                num_steps = episode_length(episode_group)
                sl = slice(offset, offset + num_steps)

                pixels_ds[sl] = episode_group["pixels"][()]
                depth_ds[sl] = episode_group["depth"][()]
                tactile_ds[sl] = episode_group["tactile"][()]
                action_ds[sl] = episode_group["action"][()]

                proprio = episode_group["proprio"][()].astype(np.float32)
                if merge_gripper and "gripper" in episode_group:
                    gripper = episode_group["gripper"][()].astype(np.float32)
                    if gripper.ndim == 1:
                        gripper = gripper[:, None]
                    proprio = np.concatenate([proprio, gripper], axis=-1)
                proprio_ds[sl] = proprio

                episode_idx_ds[sl] = episode_idx
                step_idx_ds[sl] = np.arange(num_steps, dtype=np.int64)
                env_idx_ds[sl] = env_to_idx[env_name]

                offset += num_steps

            ep_len_ds.attrs["description"] = "Length of each episode in steps."
            ep_offset_ds.attrs["description"] = "Starting global row offset for each episode."
            dst_file.attrs["source_dataset"] = str(src_path)
            dst_file.attrs["env_names_json"] = json.dumps(env_names)
            dst_file.attrs["merge_gripper_into_proprio"] = bool(merge_gripper)


def main():
    parser = argparse.ArgumentParser(
        description="Convert hierarchical Meta-World HDF5 episodes into the flat format expected by this trainer."
    )
    parser.add_argument("src", type=Path, help="Input hierarchical Meta-World HDF5 file.")
    parser.add_argument(
        "dst",
        type=Path,
        nargs="?",
        default=Path("~/.stable-wm/metaworld.h5").expanduser(),
        help="Output flat HDF5 file. Defaults to ~/.stable-wm/metaworld.h5",
    )
    parser.add_argument(
        "--keep-gripper-separate",
        action="store_true",
        help="Do not append the scalar gripper state onto the proprio vector.",
    )
    args = parser.parse_args()

    convert_dataset(
        src_path=args.src,
        dst_path=args.dst,
        merge_gripper=not args.keep_gripper_separate,
    )


if __name__ == "__main__":
    main()
