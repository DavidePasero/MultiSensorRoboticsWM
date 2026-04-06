from collections import OrderedDict

import stable_pretraining as spt
import torch
from einops import rearrange
from torch import nn

from fusion import build_fusion
from module import MLP


IMAGE_ENCODER_TYPES = {"vit", "cnn"}
IMAGE_CHANNEL_COUNTS = (1, 2, 3, 4)


def get_enabled_modality_configs(obs_cfg):
    enabled = OrderedDict()
    for name, mod_cfg in obs_cfg.modalities.items():
        if mod_cfg.get("enabled", True):
            enabled[name] = mod_cfg
    return enabled


def get_image_modality_configs(obs_cfg):
    return OrderedDict(
        (name, mod_cfg)
        for name, mod_cfg in get_enabled_modality_configs(obs_cfg).items()
        if mod_cfg.encoder_type in IMAGE_ENCODER_TYPES
    )


def _flatten_image_sequence(x: torch.Tensor):
    x = x.float()

    if x.ndim == 4:
        x = x.unsqueeze(2)

    if x.ndim != 5:
        raise ValueError(
            "Expected image tensors with shape (B, T, C, H, W), (B, T, H, W, C), "
            "or grayscale (B, T, H, W)."
        )

    b, t = x.shape[:2]
    if x.shape[2] in IMAGE_CHANNEL_COUNTS:
        flat = rearrange(x, "b t c h w -> (b t) c h w")
    elif x.shape[-1] in IMAGE_CHANNEL_COUNTS:
        flat = rearrange(x, "b t h w c -> (b t) c h w")
    else:
        raise ValueError(
            "Unable to infer channel dimension for image modality. "
            f"Got tensor shape {tuple(x.shape)}."
        )
    return flat, b, t


class BaseModalityEncoder(nn.Module):
    def __init__(self, source, output_dim):
        super().__init__()
        self.source = source
        self.output_dim = output_dim

    def get_input(self, info):
        if self.source not in info:
            raise KeyError(f"Missing observation modality '{self.source}'.")
        return info[self.source]


class ViTImageEncoder(BaseModalityEncoder):
    def __init__(
        self,
        *,
        source,
        output_dim,
        encoder_scale,
        patch_size,
        image_size,
        pretrained=False,
        projector_hidden_dim=2048,
    ):
        super().__init__(source=source, output_dim=output_dim)
        self.backbone = spt.backbone.utils.vit_hf(
            encoder_scale,
            patch_size=patch_size,
            image_size=image_size,
            pretrained=pretrained,
            use_mask_token=False,
        )

        hidden_dim = self.backbone.config.hidden_size
        self.projector = MLP(
            input_dim=hidden_dim,
            output_dim=output_dim,
            hidden_dim=projector_hidden_dim,
            norm_fn=nn.BatchNorm1d,
        )

    def forward(self, info):
        x, b, t = _flatten_image_sequence(self.get_input(info))
        output = self.backbone(x, interpolate_pos_encoding=True)
        cls_token = output.last_hidden_state[:, 0]
        emb = self.projector(cls_token)
        return rearrange(emb, "(b t) d -> b t d", b=b, t=t)


class CNNImageEncoder(BaseModalityEncoder):
    def __init__(
        self,
        *,
        source,
        output_dim,
        hidden_dims=(32, 64, 128),
        head_hidden_dim=None,
    ):
        super().__init__(source=source, output_dim=output_dim)
        hidden_dims = list(hidden_dims)
        if not hidden_dims:
            raise ValueError("CNNImageEncoder requires at least one hidden dimension.")

        layers = [
            nn.LazyConv2d(hidden_dims[0], kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        ]
        for in_dim, out_dim in zip(hidden_dims, hidden_dims[1:]):
            layers.extend(
                [
                    nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                ]
            )
        self.conv = nn.Sequential(*layers)

        last_dim = hidden_dims[-1]
        head_hidden_dim = head_hidden_dim or max(last_dim, output_dim)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_dim, head_hidden_dim),
            nn.GELU(),
            nn.Linear(head_hidden_dim, output_dim),
        )

    def forward(self, info):
        x, b, t = _flatten_image_sequence(self.get_input(info))
        x = self.conv(x)
        x = self.head(x)
        return rearrange(x, "(b t) d -> b t d", b=b, t=t)


class MLPVectorEncoder(BaseModalityEncoder):
    def __init__(self, *, source, input_dim, output_dim, hidden_dims=(128, 128)):
        super().__init__(source=source, output_dim=output_dim)
        dims = [input_dim, *list(hidden_dims), output_dim]

        layers = []
        for idx, (in_dim, out_dim) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if idx < len(dims) - 2:
                layers.extend([nn.LayerNorm(out_dim), nn.GELU()])

        self.net = nn.Sequential(*layers)

    def forward(self, info):
        x = self.get_input(info).float()
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        if x.ndim != 3:
            raise ValueError(
                "Expected vector observations with shape (B, T, D). "
                f"Got tensor shape {tuple(x.shape)}."
            )

        b, t = x.shape[:2]
        x = rearrange(x, "b t d -> (b t) d")
        x = self.net(x)
        return rearrange(x, "(b t) d -> b t d", b=b, t=t)


class MultiModalObsEncoder(nn.Module):
    is_obs_encoder = True

    def __init__(
        self,
        encoders,
        fusion,
        *,
        keep_modality_embeddings=False,
        primary_source="pixels",
        hidden_dim=None,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.fusion = fusion
        self.keep_modality_embeddings = keep_modality_embeddings
        self.primary_source = primary_source
        self.hidden_dim = hidden_dim

    def forward(self, info):
        modality_embs = OrderedDict(
            (name, encoder(info)) for name, encoder in self.encoders.items()
        )
        fused_emb, aux = self.fusion(modality_embs)

        output = {"emb": fused_emb}
        if self.keep_modality_embeddings:
            output["modality_embs"] = modality_embs
        output.update(aux)
        return output


def build_modality_encoder(cfg, name, mod_cfg):
    source = mod_cfg.get("source", name)
    encoder_type = mod_cfg.encoder_type
    output_dim = mod_cfg.output_dim

    if encoder_type == "vit":
        return ViTImageEncoder(
            source=source,
            output_dim=output_dim,
            encoder_scale=mod_cfg.get("encoder_scale", cfg.encoder_scale),
            patch_size=mod_cfg.get("patch_size", cfg.patch_size),
            image_size=mod_cfg.get("img_size", cfg.img_size),
            pretrained=mod_cfg.get("pretrained", False),
            projector_hidden_dim=mod_cfg.get("projector_hidden_dim", 2048),
        )

    if encoder_type == "cnn":
        return CNNImageEncoder(
            source=source,
            output_dim=output_dim,
            hidden_dims=mod_cfg.get("hidden_dims", (32, 64, 128)),
            head_hidden_dim=mod_cfg.get("head_hidden_dim"),
        )

    if encoder_type == "mlp":
        input_dim = mod_cfg.get("input_dim")
        if input_dim is None:
            input_dim = getattr(cfg.wm, f"{source}_dim", None)
        if input_dim is None:
            raise ValueError(
                f"Missing input_dim for vector modality '{source}'. "
                "Make sure its dataset column is loaded so train.py can infer it."
            )

        return MLPVectorEncoder(
            source=source,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=mod_cfg.get("hidden_dims", (128, 128)),
        )

    raise ValueError(f"Unsupported modality encoder type: {encoder_type}")


def build_obs_encoder(cfg):
    modality_cfgs = get_enabled_modality_configs(cfg.obs_encoder)
    if not modality_cfgs:
        raise ValueError("At least one observation modality must be enabled.")

    encoders = OrderedDict()
    input_dims = OrderedDict()
    primary_source = None
    hidden_dim = cfg.wm.embed_dim

    for name, mod_cfg in modality_cfgs.items():
        source = mod_cfg.get("source", name)
        encoders[name] = build_modality_encoder(cfg, name, mod_cfg)
        input_dims[name] = encoders[name].output_dim

        if primary_source is None or source == "pixels":
            primary_source = source

        backbone = getattr(encoders[name], "backbone", None)
        if backbone is not None and hasattr(backbone, "config"):
            hidden_dim = getattr(backbone.config, "hidden_size", hidden_dim)

    fusion = build_fusion(cfg.obs_encoder.fusion, input_dims)

    return MultiModalObsEncoder(
        encoders=encoders,
        fusion=fusion,
        keep_modality_embeddings=cfg.obs_encoder.get("keep_modality_embeddings", False),
        primary_source=primary_source,
        hidden_dim=hidden_dim,
    )
