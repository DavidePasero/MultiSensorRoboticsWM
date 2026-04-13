import sys
from pathlib import Path

import h5py
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DATASETS_DIR = REPO_ROOT / "datasets"
if str(DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASETS_DIR))

from datasets_utils.dataset_factory import get_dataset_adapter_from_config
from multimodal import get_enabled_modality_configs


def resolve_checkpoint_path(checkpoint: str, cache_dir: Path | None) -> Path:
    path = Path(checkpoint).expanduser()
    if path.exists():
        if path.is_dir():
            candidates = sorted(path.glob("*_object.ckpt"))
            if not candidates:
                raise FileNotFoundError(f"No *_object.ckpt found in {path}")
            return candidates[-1]
        if path.is_file():
            if path.name.endswith("_object.ckpt"):
                return path
            raise ValueError(
                f"Checkpoint file must end with '_object.ckpt', got {path.name}."
            )

    root = cache_dir or Path(swm.data.utils.get_cache_dir())
    run_path = Path(root, checkpoint)
    if run_path.is_dir():
        candidates = sorted(run_path.glob("*_object.ckpt"))
        if not candidates:
            raise FileNotFoundError(f"No *_object.ckpt found in {run_path}")
        return candidates[-1]

    path = Path(f"{run_path}_object.ckpt")
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    return path


def load_cost_model(checkpoint: str, cache_dir: Path | None):
    path = resolve_checkpoint_path(checkpoint, cache_dir)
    model = torch.load(path, weights_only=False, map_location="cpu")

    if hasattr(model, "get_cost"):
        return model

    def scan_module(module):
        if hasattr(module, "get_cost"):
            return module
        for child in module.children():
            result = scan_module(child)
            if result is not None:
                return result
        return None

    result = scan_module(model)
    if result is None:
        raise RuntimeError(f"No module with 'get_cost' found in {path}")
    return result


def infer_checkpoint_dir(checkpoint: str, cache_dir: Path | None) -> Path:
    path = Path(checkpoint).expanduser()

    if path.exists():
        if path.is_dir():
            return path
        if path.name.endswith("_object.ckpt"):
            return path.parent
        return path.parent

    root = cache_dir or Path(swm.data.utils.get_cache_dir())
    run_path = Path(root, checkpoint)
    return run_path if run_path.is_dir() else run_path.parent


def infer_config_path(checkpoint: str, cache_dir: Path | None) -> Path:
    checkpoint_dir = infer_checkpoint_dir(checkpoint, cache_dir)
    candidates = [
        checkpoint_dir / "config.yaml",
        Path(cache_dir or swm.data.utils.get_cache_dir()) / "config.yaml",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not infer config.yaml. Pass --config explicitly."
    )


def load_experiment_config(
    checkpoint: str,
    cache_dir: Path | None,
    config_path: Path | None = None,
    dataset_name: str | None = None,
):
    resolved_config_path = (
        config_path.expanduser()
        if config_path is not None
        else infer_config_path(checkpoint, cache_dir)
    )
    cfg = OmegaConf.load(resolved_config_path)
    if dataset_name is not None:
        cfg.data.dataset.name = dataset_name
    return cfg, resolved_config_path


def get_dataset_path(dataset_name: str, cache_dir: Path | None = None) -> Path:
    return Path(cache_dir or swm.data.utils.get_cache_dir(), f"{dataset_name}.h5")


def get_dataset_columns(dataset_name: str, cache_dir: Path | None = None) -> list[str]:
    dataset_path = get_dataset_path(dataset_name, cache_dir)
    with h5py.File(dataset_path, "r") as f:
        return sorted(k for k in f.keys() if k not in ("ep_len", "ep_offset"))


def build_dataset(
    cfg,
    cache_dir: Path | None,
    extra_keys_to_load: list[str] | None = None,
    passthrough_keys: list[str] | None = None,
):
    dataset_adapter = get_dataset_adapter_from_config(cfg)
    dataset, _ = dataset_adapter.build_dataset(
        cfg,
        cache_dir=cache_dir,
        extra_keys_to_load=extra_keys_to_load,
        passthrough_keys=passthrough_keys,
    )
    return dataset


def get_modalities(cfg, dataset, selected=None):
    enabled = list(get_enabled_modality_configs(cfg.obs_encoder).keys())
    if selected is None:
        return [name for name in enabled if name in dataset.column_names]

    unknown = [name for name in selected if name not in enabled]
    if unknown:
        raise ValueError(f"Unknown modalities requested: {unknown}")

    return [name for name in selected if name in dataset.column_names]


def copy_batch(batch):
    return {
        k: v.clone() if torch.is_tensor(v) else v
        for k, v in batch.items()
    }


def batch_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


def extract_representation(
    model,
    batch,
    *,
    representation: str = "fused",
    probe_step: int = -1,
):
    batch = copy_batch(batch)
    if "action" in batch:
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    encoder = getattr(model, "encoder", None)
    previous_keep = None
    if (
        representation != "fused"
        and encoder is not None
        and hasattr(encoder, "keep_modality_embeddings")
    ):
        previous_keep = encoder.keep_modality_embeddings
        encoder.keep_modality_embeddings = True

    try:
        outputs = model.encode(batch)
        if representation == "fused":
            rep = outputs["emb"]
        else:
            modality_embs = outputs.get("modality_embs")
            if modality_embs is None:
                raise ValueError(
                    "Model did not expose modality embeddings. "
                    "Set keep_modality_embeddings=True before probing modality branches."
                )
            if representation not in modality_embs:
                raise KeyError(
                    f"Representation '{representation}' not found in modality embeddings: "
                    f"{list(modality_embs.keys())}"
                )
            rep = modality_embs[representation]
    finally:
        if previous_keep is not None:
            encoder.keep_modality_embeddings = previous_keep

    return rep[:, probe_step]
