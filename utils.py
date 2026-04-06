import numpy as np
import torch
from pathlib import Path
import torch.nn.functional as F
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

IMAGE_CHANNEL_COUNTS = (1, 2, 3, 4)


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def _prepare_image_sequence(x):
    x = torch.as_tensor(x)

    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
        had_sequence = False
    elif x.ndim == 3:
        had_sequence = False
        if x.shape[0] in IMAGE_CHANNEL_COUNTS:
            x = x.unsqueeze(0)
        elif x.shape[-1] in IMAGE_CHANNEL_COUNTS:
            x = x.permute(2, 0, 1).unsqueeze(0)
        else:
            x = x.unsqueeze(1)
            had_sequence = True
    elif x.ndim == 4:
        had_sequence = True
        if x.shape[1] in IMAGE_CHANNEL_COUNTS:
            pass
        elif x.shape[-1] in IMAGE_CHANNEL_COUNTS:
            x = x.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unable to infer channels for image tensor with shape {tuple(x.shape)}.")
    else:
        raise ValueError(f"Expected image tensor with 2 to 4 dims, got shape {tuple(x.shape)}.")

    return x.float(), had_sequence


def get_image_like_preprocessor(
    source: str,
    target: str,
    img_size: int,
    mean=None,
    std=None,
    expect_sequence=True,
):
    def preprocess(x):
        x = torch.as_tensor(x)

        if expect_sequence:
            if x.ndim == 2:
                x = x.unsqueeze(0).unsqueeze(0)
                had_sequence = False
            elif x.ndim == 3:
                # Interpret 3D tensors as grayscale sequences: (T, H, W).
                x = x.unsqueeze(1)
                had_sequence = True
            elif x.ndim == 4:
                had_sequence = True
                if x.shape[1] in IMAGE_CHANNEL_COUNTS:
                    pass
                elif x.shape[-1] in IMAGE_CHANNEL_COUNTS:
                    x = x.permute(0, 3, 1, 2)
                else:
                    raise ValueError(
                        f"Unable to infer channels for image sequence with shape {tuple(x.shape)}."
                    )
            else:
                raise ValueError(
                    f"Expected image sequence tensor with 2 to 4 dims, got shape {tuple(x.shape)}."
                )
            x = x.float()
        else:
            x, had_sequence = _prepare_image_sequence(x)

        if x.numel() and x.max() > 1:
            x = x / 255.0

        if img_size is not None and x.shape[-2:] != (img_size, img_size):
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)

        if mean is not None and std is not None:
            mean_t = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
            std_t = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)

            if mean_t.size(1) == 1 and x.size(1) != 1:
                mean_t = mean_t.expand(1, x.size(1), 1, 1)
                std_t = std_t.expand(1, x.size(1), 1, 1)
            elif mean_t.size(1) != x.size(1):
                raise ValueError(
                    f"Normalization stats expect {mean_t.size(1)} channels, got {x.size(1)}."
                )

            x = (x - mean_t) / std_t.clamp_min(1e-6)

        return x if had_sequence else x[0]

    return dt.transforms.WrapTorchTransform(preprocess, source=source, target=target)


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()

    def norm_fn(x):
        return ((x - mean) / std).float()

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")
