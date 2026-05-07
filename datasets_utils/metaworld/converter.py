"""Meta-World converter from hierarchical rollout HDF5 files to the flat training format."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from datasets_utils.metaworld.preprocessor import (
    finalize_running_vector_stats,
    init_running_vector_stats,
    update_running_vector_stats,
    write_vector_stats,
)


EE_POSITION_SOURCE_KEYS = ("ee_position", "ee_xyz")
IMAGE_CHANNEL_COUNTS = (1, 2, 3, 4)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_PIXELS_SIZE = 224
DEFAULT_DEPTH_SIZE = 224
DEFAULT_TACTILE_SIZE = 64


def iter_episodes(src_file):
    for env_name in sorted(src_file.keys()):
        env_group = src_file[env_name]
        for episode_name in sorted(env_group.keys()):
            yield env_name, episode_name, env_group[episode_name]


def require_episode_key(episode_group, key: str):
    if key not in episode_group:
        raise KeyError(
            f"Episode group is missing required key '{key}'. "
            f"Available keys: {sorted(episode_group.keys())}"
        )
    return episode_group[key]


def episode_length(episode_group):
    return int(require_episode_key(episode_group, "action").shape[0])


def find_first_present_key(episode_group, candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        if key in episode_group:
            return key
    return None


def _prepare_image_sequence(values):
    values = torch.as_tensor(values)
    if values.ndim == 3:
        values = values.unsqueeze(1)
    elif values.ndim == 4:
        if values.shape[1] in IMAGE_CHANNEL_COUNTS:
            pass
        elif values.shape[-1] in IMAGE_CHANNEL_COUNTS:
            values = values.permute(0, 3, 1, 2)
        else:
            raise ValueError(
                "Unable to infer channels for image sequence with shape "
                f"{tuple(values.shape)}."
            )
    else:
        raise ValueError(
            f"Expected image sequence tensor with 3 or 4 dims, got {tuple(values.shape)}."
        )
    return values.float()


def preprocess_image_sequence(values, img_size, mean=None, std=None):
    values = _prepare_image_sequence(values)

    if values.numel() and values.amax() > 1:
        values = values / 255.0

    if img_size is not None and values.shape[-2:] != (img_size, img_size):
        values = F.interpolate(
            values,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, dtype=values.dtype).view(1, -1, 1, 1)
        std_t = torch.tensor(std, dtype=values.dtype).view(1, -1, 1, 1).clamp_min(1e-6)
        values = (values - mean_t) / std_t

    return values.cpu().numpy().astype(np.float16, copy=False)


def _merge_proprio_and_gripper(episode_group, merge_gripper):
    proprio = require_episode_key(episode_group, "proprio")[()].astype(np.float32)
    gripper = None
    if "gripper" in episode_group:
        gripper = require_episode_key(episode_group, "gripper")[()].astype(np.float32)
        if gripper.ndim == 1:
            gripper = gripper[:, None]

    if merge_gripper and gripper is not None:
        proprio = np.concatenate([proprio, gripper], axis=-1)

    return proprio, gripper


def infer_shapes(src_file, merge_gripper):
    for _env_name, _episode_name, episode_group in iter_episodes(src_file):
        pixels = preprocess_image_sequence(
            require_episode_key(episode_group, "pixels")[()],
            img_size=DEFAULT_PIXELS_SIZE,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
        depth = preprocess_image_sequence(
            require_episode_key(episode_group, "depth")[()],
            img_size=DEFAULT_DEPTH_SIZE,
        )
        tactile = preprocess_image_sequence(
            require_episode_key(episode_group, "tactile")[()],
            img_size=DEFAULT_TACTILE_SIZE,
        )
        proprio, gripper = _merge_proprio_and_gripper(episode_group, merge_gripper)
        shapes = {
            "pixels": pixels.shape[1:],
            "depth": depth.shape[1:],
            "tactile": tactile.shape[1:],
            "proprio": proprio.shape[1:],
            "action": require_episode_key(episode_group, "action").shape[1:],
            "force_torque": require_episode_key(
                episode_group, "force_torque"
            ).shape[1:],
        }

        if gripper is not None:
            shapes["gripper"] = gripper.shape[1:]

        ee_position_source = find_first_present_key(
            episode_group, EE_POSITION_SOURCE_KEYS
        )
        if ee_position_source is not None:
            shapes["ee_position"] = require_episode_key(
                episode_group, ee_position_source
            ).shape[1:]

        for key in ("object_1_xyz", "object_2_xyz", "bool_contact"):
            if key in episode_group:
                shapes[key] = episode_group[key].shape[1:]
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
        first_episode_group = episodes[0][2]
        ee_position_source = find_first_present_key(
            first_episode_group, EE_POSITION_SOURCE_KEYS
        )
        force_torque_stats = init_running_vector_stats(shapes["force_torque"][0])

        with h5py.File(dst_path, "w") as dst_file:
            pixels_ds = create_dataset(
                dst_file, "pixels", total_steps, shapes["pixels"], np.float16
            )
            depth_ds = create_dataset(
                dst_file, "depth", total_steps, shapes["depth"], np.float16
            )
            tactile_ds = create_dataset(
                dst_file, "tactile", total_steps, shapes["tactile"], np.float16
            )
            proprio_ds = create_dataset(
                dst_file, "proprio", total_steps, shapes["proprio"], np.float32
            )
            force_torque_ds = create_dataset(
                dst_file,
                "force_torque",
                total_steps,
                shapes["force_torque"],
                np.float32,
            )
            action_ds = create_dataset(
                dst_file, "action", total_steps, shapes["action"], np.float32
            )
            gripper_ds = None
            if "gripper" in shapes:
                gripper_ds = create_dataset(
                    dst_file, "gripper", total_steps, shapes["gripper"], np.float32
                )
            ee_position_ds = None
            if "ee_position" in shapes:
                ee_position_ds = create_dataset(
                    dst_file,
                    "ee_position",
                    total_steps,
                    shapes["ee_position"],
                    np.float32,
                )
            object_1_xyz_ds = None
            if "object_1_xyz" in shapes:
                object_1_xyz_ds = create_dataset(
                    dst_file,
                    "object_1_xyz",
                    total_steps,
                    shapes["object_1_xyz"],
                    np.float32,
                )
            object_2_xyz_ds = None
            if "object_2_xyz" in shapes:
                object_2_xyz_ds = create_dataset(
                    dst_file,
                    "object_2_xyz",
                    total_steps,
                    shapes["object_2_xyz"],
                    np.float32,
                )
            bool_contact_ds = None
            if "bool_contact" in shapes:
                bool_contact_ds = create_dataset(
                    dst_file,
                    "bool_contact",
                    total_steps,
                    shapes["bool_contact"],
                    np.bool_,
                )
            ep_len_ds = dst_file.create_dataset(
                "ep_len",
                data=episode_lengths,
                dtype=np.int64,
            )
            ep_offset_ds = dst_file.create_dataset(
                "ep_offset",
                data=episode_offsets,
                dtype=np.int64,
            )
            episode_idx_ds = dst_file.create_dataset(
                "episode_idx",
                shape=(total_steps,),
                dtype=np.int64,
            )
            step_idx_ds = dst_file.create_dataset(
                "step_idx",
                shape=(total_steps,),
                dtype=np.int64,
            )
            env_idx_ds = dst_file.create_dataset(
                "env_idx",
                shape=(total_steps,),
                dtype=np.int64,
            )

            offset = 0
            for episode_idx, (env_name, _episode_name, episode_group) in enumerate(episodes):
                num_steps = episode_length(episode_group)
                sl = slice(offset, offset + num_steps)

                pixels_ds[sl] = preprocess_image_sequence(
                    require_episode_key(episode_group, "pixels")[()],
                    img_size=DEFAULT_PIXELS_SIZE,
                    mean=IMAGENET_MEAN,
                    std=IMAGENET_STD,
                )
                depth_ds[sl] = preprocess_image_sequence(
                    require_episode_key(episode_group, "depth")[()],
                    img_size=DEFAULT_DEPTH_SIZE,
                )
                tactile_ds[sl] = preprocess_image_sequence(
                    require_episode_key(episode_group, "tactile")[()],
                    img_size=DEFAULT_TACTILE_SIZE,
                )

                action_ds[sl] = require_episode_key(episode_group, "action")[()].astype(
                    np.float32
                )

                force_torque = require_episode_key(
                    episode_group, "force_torque"
                )[()].astype(np.float32)
                force_torque_ds[sl] = force_torque
                update_running_vector_stats(force_torque_stats, force_torque)

                proprio, gripper = _merge_proprio_and_gripper(episode_group, merge_gripper)
                proprio_ds[sl] = proprio
                if gripper_ds is not None and gripper is not None:
                    gripper_ds[sl] = gripper

                if ee_position_ds is not None:
                    source_key = find_first_present_key(
                        episode_group, EE_POSITION_SOURCE_KEYS
                    )
                    if source_key is None:
                        raise KeyError(
                            "Expected one of "
                            f"{list(EE_POSITION_SOURCE_KEYS)} in episode group, "
                            f"available keys: {sorted(episode_group.keys())}"
                        )
                    ee_position_ds[sl] = require_episode_key(
                        episode_group, source_key
                    )[()].astype(np.float32)

                if object_1_xyz_ds is not None:
                    object_1_xyz_ds[sl] = require_episode_key(
                        episode_group, "object_1_xyz"
                    )[()].astype(np.float32)
                if object_2_xyz_ds is not None:
                    object_2_xyz_ds[sl] = require_episode_key(
                        episode_group, "object_2_xyz"
                    )[()].astype(np.float32)
                if bool_contact_ds is not None:
                    bool_contact_ds[sl] = require_episode_key(
                        episode_group, "bool_contact"
                    )[()].astype(np.bool_)

                episode_idx_ds[sl] = episode_idx
                step_idx_ds[sl] = np.arange(num_steps, dtype=np.int64)
                env_idx_ds[sl] = env_to_idx[env_name]

                offset += num_steps

            force_torque_mean, force_torque_std = finalize_running_vector_stats(
                force_torque_stats
            )
            write_vector_stats(
                dst_file,
                column="force_torque",
                mean=force_torque_mean,
                std=force_torque_std,
            )

            ep_len_ds.attrs["description"] = "Length of each episode in steps."
            ep_offset_ds.attrs["description"] = "Starting global row offset for each episode."
            dst_file.attrs["source_dataset"] = str(src_path)
            dst_file.attrs["env_names_json"] = json.dumps(env_names)
            dst_file.attrs["merge_gripper_into_proprio"] = bool(merge_gripper)
            dst_file.attrs["preprocessed"] = True
            dst_file.attrs["images_preprocessed"] = True
            dst_file.attrs["pixels_size"] = DEFAULT_PIXELS_SIZE
            dst_file.attrs["depth_size"] = DEFAULT_DEPTH_SIZE
            dst_file.attrs["tactile_size"] = DEFAULT_TACTILE_SIZE
            if ee_position_source is not None:
                dst_file.attrs["ee_position_source_key"] = ee_position_source


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Convert hierarchical Meta-World HDF5 episodes into the flat format "
            "expected by this trainer."
        )
    )
    parser.add_argument("src", type=Path, help="Input hierarchical Meta-World HDF5 file.")
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


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    convert_dataset(
        src_path=args.src,
        dst_path=args.dst,
        merge_gripper=not args.keep_gripper_separate,
    )


if __name__ == "__main__":
    main()
