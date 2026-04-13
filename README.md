
# LeWorldModel
### Stable End-to-End Joint-Embedding Predictive Architecture from Pixels

[Lucas Maes*](https://x.com/lucasmaes_), [Quentin Le Lidec*](https://quentinll.github.io/), [Damien Scieur](https://scholar.google.com/citations?user=hNscQzgAAAAJ&hl=fr), [Yann LeCun](https://yann.lecun.com/) and [Randall Balestriero](https://randallbalestriero.github.io/)

**Abstract:** Joint Embedding Predictive Architectures (JEPAs) offer a compelling framework for learning world models in compact latent spaces, yet existing methods remain fragile, relying on complex multi-term losses, exponential moving averages, pretrained encoders, or auxiliary supervision to avoid representation collapse. In this work, we introduce LeWorldModel (LeWM), the first JEPA that trains stably end-to-end from raw pixels using only two loss terms: a next-embedding prediction loss and a regularizer enforcing Gaussian-distributed latent embeddings. This reduces tunable loss hyperparameters from six to one compared to the only existing end-to-end alternative. With ~15M parameters trainable on a single GPU in a few hours, LeWM plans up to 48× faster than foundation-model-based world models while remaining competitive across diverse 2D and 3D control tasks. Beyond control, we show that LeWM's latent space encodes meaningful physical structure through probing of physical quantities. Surprise evaluation confirms that the model reliably detects physically implausible events.

<p align="center">
   <b>[ <a href="https://arxiv.org/pdf/2603.19312v1">Paper</a> | <a href="https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e?usp=sharing">Checkpoints</a> | <a href="https://huggingface.co/collections/quentinll/lewm">Data</a> | <a href="https://le-wm.github.io/">Website</a> ]</b>
</p>

<br>

<p align="center">
  <img src="assets/lewm.gif" width="80%">
</p>

If you find this code useful, please reference it in your paper:
```
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

## Using the code
This codebase builds on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) for environment management, planning, and evaluation, and [stable-pretraining](https://github.com/galilai-group/stable-pretraining) for training. Together they reduce this repository to its core contribution: the model architecture and training objective.

**Installation:**
```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Data

Datasets use the HDF5 format for fast loading. Download the data from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress with:

```bash
tar --zstd -xvf archive.tar.zst
```

Place the extracted `.h5` files under `$STABLEWM_HOME` (defaults to `~/.stable_worldmodel/`). You can override this path:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Dataset names are specified without the `.h5` extension. For example, `config/train/data/pusht.yaml` references `pusht_expert_train`, which resolves to `$STABLEWM_HOME/pusht_expert_train.h5`.

## Training

`jepa.py` contains the PyTorch implementation of LeWM. Training is configured via [Hydra](https://hydra.cc/) config files under `config/train/`.

Before training, set your WandB `entity` and `project` in `config/train/lewm.yaml`:
```yaml
wandb:
  config:
    entity: your_entity
    project: your_project
```

To launch training:
```bash
python train.py data=pusht
```

Checkpoints are saved to `$STABLEWM_HOME` upon completion.

For baseline scripts, see the stable-worldmodel [scripts](https://github.com/galilai-group/stable-worldmodel/tree/main/scripts/train) folder.

## Multimodal Training

The repository now includes a modular observation encoder path for multimodal training. The main training loop still optimizes the same JEPA objective and uses the same action encoder, predictor, and `SIGReg` loss, but observation encoding is no longer hardcoded to RGB pixels.

Two files define the multimodal stack:
- `multimodal.py` contains modality-specific encoders and the top-level observation encoder
- `fusion.py` contains fusion modules

The default pixel-only setup remains available through:
```bash
python train.py data=pusht obs_encoder=pixels
```

The current multimodal baseline is configured in `config/train/obs_encoder/multimodal.yaml` and includes:
- `pixels`: ViT encoder with ImageNet preprocessing
- `depth`: lightweight CNN encoder with generic image preprocessing
- `tactile`: lightweight CNN encoder with generic image preprocessing
- `proprio`: MLP encoder with per-dimension normalization
- `force_torque`: MLP encoder with dataset-wide normalization loaded from saved HDF5 stats

The modality embeddings are concatenated and projected back to `wm.embed_dim` before being passed to the unchanged JEPA predictor. The fusion abstraction is intentionally separate so additional implementations such as cross-attention or MoE can be added later without changing `train.py` or `jepa.py`.

## Meta-World Conversion

The `stable_worldmodel` HDF5 loader expects a flat dataset with root-level arrays such as `pixels`, `action`, `ep_len`, and `ep_offset`. A raw Meta-World collection file produced by per-environment/per-episode groups does not match that format directly.

Dataset-specific preprocessing and conversion code now lives under `datasets/`. The training config declares the dataset family through `data.dataset.type`, and the code instantiates the matching dataset adapter automatically.

The current built-in dataset families are:
- `generic`: flat HDF5 datasets that use the default image preprocessing and per-column z-score normalization
- `metaworld`: flat HDF5 datasets converted from hierarchical Meta-World rollouts, with saved `force_torque` normalization stats

For Meta-World, the relevant implementation lives under `datasets/metaworld/`. The Meta-World converter:
- flattens all episodes into root-level arrays
- writes `ep_len` and `ep_offset` for episode indexing
- writes `episode_idx` and `step_idx` for bookkeeping
- merges `gripper` into `proprio` by default
- preserves `pixels`, `depth`, `tactile`, `proprio`, `force_torque`, and `action`
- saves dataset-wide normalization statistics for `force_torque` under `stats/force_torque_mean` and `stats/force_torque_std`

Convert a raw dataset with:
```bash
python datasets/convert_dataset.py metaworld /path/to/raw_metaworld.hdf5 ~/.stable_worldmodel/metaworld.h5
```

The corresponding training config is `config/train/data/metaworld.yaml`, which expects the dataset name `metaworld` and uses `frameskip: 1`.

To train on the converted dataset:
```bash
export STABLEWM_HOME=~/.stable_worldmodel
python train.py data=metaworld obs_encoder=multimodal
```

If you want to keep `gripper` separate instead of appending it to `proprio`, convert with:
```bash
python datasets/convert_dataset.py metaworld /path/to/raw_metaworld.hdf5 ~/.stable_worldmodel/metaworld.h5 --keep-gripper-separate
```

In that case you should also update the multimodal config to add a dedicated encoder branch for `gripper`.

## Planning

Evaluation configs live under `config/eval/`. Set the `policy` field to the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix:

```bash
# ✓ correct
python eval.py --config-name=pusht.yaml policy=pusht/lewm

# ✗ incorrect
python eval.py --config-name=pusht.yaml policy=pusht/lewm_object.ckpt
```

## Pretrained Checkpoints

Pre-trained checkpoints are available on [Google Drive](https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e). Download the checkpoint archive and place the extracted files under `$STABLEWM_HOME/`.

<div align="center">

| Method | two-room | pusht | cube | reacher |
|:---:|:---:|:---:|:---:|:---:|
| pldm | ✓ | ✓ | ✓ | ✓ |
| lejepa | ✓ | ✓ | ✓ | ✓ |
| ivl | ✓ | ✓ | ✓ | — |
| iql | ✓ | ✓ | ✓ | — |
| gcbc | ✓ | ✓ | ✓ | — |
| dinowm | ✓ | ✓ | — | — |
| dinowm_noprop | ✓ | ✓ | ✓ | ✓ |

</div>

## Loading a checkpoint

Each tar archive contains two files per checkpoint:
- `<name>_object.ckpt` — a serialized Python object for convenient loading; this is what `eval.py` and the `stable_worldmodel` API use
- `<name>_weight.ckpt` — a weights-only checkpoint (`state_dict`) for cases where you want to load weights into your own model instance

To load the object checkpoint via the `stable_worldmodel` API:

```python
import stable_worldmodel as swm

# Load the cost model (for MPC)
cost = swm.policy.AutoCostModel('pusht/lewm')
```

This function accepts:
- `run_name` — checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix
- `cache_dir` — optional override for the checkpoint root (defaults to `$STABLEWM_HOME`)

The returned module is in `eval` mode with its PyTorch weights accessible via `.state_dict()`.

## Contact & Contributions
Feel free to open [issues](https://github.com/lucas-maes/le-wm/issues)! For questions or collaborations, please contact `lucas.maes@mila.quebec`
