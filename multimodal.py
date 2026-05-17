from collections import OrderedDict

import stable_pretraining as spt
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from fusion import build_fusion
from imputer import build_imputer
from module import MLP


IMAGE_ENCODER_TYPES = {"vit", "cnn"}
VECTOR_ENCODER_TYPES = {"mlp"}
IMAGE_CHANNEL_COUNTS = (1, 2, 3, 4)
DEFAULT_MODALITY_CHANNELS = {
    "pixels": 3,
    "depth": 1,
    "tactile": 2,
}


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


def get_vector_modality_configs(obs_cfg):
    return OrderedDict(
        (name, mod_cfg)
        for name, mod_cfg in get_enabled_modality_configs(obs_cfg).items()
        if mod_cfg.encoder_type in VECTOR_ENCODER_TYPES
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


def _preprocess_image_sequence(x: torch.Tensor, *, img_size=None, mean=None, std=None):
    x, b, t = _flatten_image_sequence(x)

    if x.numel() and x.amax() > 1:
        x = x / 255.0

    if img_size is not None and x.shape[-2:] != (img_size, img_size):
        x = F.interpolate(
            x,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    if mean is not None and std is not None:
        mean = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        std = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        std = std.clamp_min(1e-6)

        if mean.size(1) == 1 and x.size(1) != 1:
            mean = mean.expand(1, x.size(1), 1, 1)
            std = std.expand(1, x.size(1), 1, 1)
        elif mean.size(1) != x.size(1):
            raise ValueError(
                f"Normalization stats expect {mean.size(1)} channels, got {x.size(1)}."
            )

        x = (x - mean) / std

    return x, b, t


def _default_image_preprocess(source, *, img_size=None):
    if img_size is None:
        if source == "tactile":
            img_size = 64
        elif source in {"pixels", "depth"}:
            img_size = 224

    if source == "pixels":
        imagenet_stats = spt.data.dataset_stats.ImageNet
        mean = imagenet_stats["mean"]
        std = imagenet_stats["std"]
    else:
        mean = None
        std = None

    return img_size, mean, std


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
        image_size = getattr(self.backbone.config, "image_size", None)
        image_size, mean, std = _default_image_preprocess(
            self.source,
            img_size=image_size,
        )
        x, b, t = _preprocess_image_sequence(
            self.get_input(info),
            img_size=image_size,
            mean=mean,
            std=std,
        )
        output = self.backbone(x, interpolate_pos_encoding=True)
        cls_token = output.last_hidden_state[:, 0]
        emb = self.projector(cls_token)
        return rearrange(emb, "(b t) d -> b t d", b=b, t=t)


class CNNImageEncoder(BaseModalityEncoder):
    def __init__(
        self,
        *,
        source,
        in_channels,
        output_dim,
        hidden_dims=(32, 64, 128),
        head_hidden_dim=None,
    ):
        super().__init__(source=source, output_dim=output_dim)
        hidden_dims = list(hidden_dims)
        if not hidden_dims:
            raise ValueError("CNNImageEncoder requires at least one hidden dimension.")
        if in_channels is None:
            raise ValueError(
                f"Missing in_channels for CNN image modality '{source}'."
            )

        layers = [
            nn.Conv2d(
                in_channels,
                hidden_dims[0],
                kernel_size=5,
                stride=2,
                padding=2,
            ),
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
        img_size, mean, std = _default_image_preprocess(self.source)
        x, b, t = _preprocess_image_sequence(
            self.get_input(info),
            img_size=img_size,
            mean=mean,
            std=std,
        )
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
        imputer=None,
        keep_modality_embeddings=False,
        primary_source="pixels",
        hidden_dim=None,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.imputer = imputer
        self.fusion = fusion
        self.keep_modality_embeddings = keep_modality_embeddings
        self.primary_source = primary_source
        self.hidden_dim = hidden_dim

    def forward(self, info):
        modality_embs = OrderedDict()
        missing = []
        imputer = getattr(self, "imputer", None)
        for name, encoder in self.encoders.items():
            if encoder.source not in info:
                missing.append(name)
                continue
            modality_embs[name] = encoder(info)

        if not modality_embs:
            expected = [encoder.source for encoder in self.encoders.values()]
            raise KeyError(
                "No observation modalities were available for encoding. "
                f"Expected one of: {expected}."
            )

        aux = {}
        fusion_inputs = modality_embs
        if imputer is not None:
            if missing and not getattr(
                imputer, "supports_missing_modalities", False
            ):
                raise KeyError(
                    "Missing observation modalities for the selected imputer: "
                    f"{missing}."
                )
            aux = imputer(modality_embs)
            fusion_inputs = aux.pop("modality_tokens")
        elif missing and not getattr(self.fusion, "supports_missing_modalities", False):
            raise KeyError(
                "Missing observation modalities for the selected fusion module: "
                f"{missing}."
            )

        fused_emb, fusion_aux = self.fusion(fusion_inputs)

        output = {"emb": fused_emb}
        if self.keep_modality_embeddings:
            output["modality_embs"] = modality_embs
            if imputer is not None:
                output["modality_tokens"] = fusion_inputs
        output.update(aux)
        output.update(fusion_aux)
        return output


def build_modality_encoder(cfg, name, mod_cfg):
    source = mod_cfg.get("source", name)
    encoder_type = mod_cfg.encoder_type
    output_dim = mod_cfg.output_dim

    if encoder_type == "vit":
        preprocess = mod_cfg.get(
            "preprocess",
            "imagenet" if source == "pixels" else "generic",
        )
        img_size = mod_cfg.get("img_size", cfg.img_size)
        if preprocess not in {"imagenet", "generic"}:
            raise ValueError(
                f"Unsupported preprocess type '{preprocess}' for source '{source}'."
            )

        return ViTImageEncoder(
            source=source,
            output_dim=output_dim,
            encoder_scale=mod_cfg.get("encoder_scale", cfg.encoder_scale),
            patch_size=mod_cfg.get("patch_size", cfg.patch_size),
            image_size=img_size,
            pretrained=mod_cfg.get("pretrained", False),
            projector_hidden_dim=mod_cfg.get("projector_hidden_dim", 2048),
        )

    if encoder_type == "cnn":
        in_channels = mod_cfg.get("in_channels")
        if in_channels is None:
            in_channels = DEFAULT_MODALITY_CHANNELS.get(source)

        preprocess = mod_cfg.get(
            "preprocess",
            "imagenet" if source == "pixels" else "generic",
        )
        if preprocess not in {"imagenet", "generic"}:
            raise ValueError(
                f"Unsupported preprocess type '{preprocess}' for source '{source}'."
            )

        return CNNImageEncoder(
            source=source,
            in_channels=in_channels,
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

    imputer = None
    fusion_input_dims = input_dims
    if cfg.obs_encoder.get("imputer") is not None:
        imputer = build_imputer(
            cfg.obs_encoder,
            input_dims,
            default_model_dim=cfg.wm.embed_dim,
        )
        fusion_input_dims = OrderedDict(
            (name, imputer.model_dim) for name in input_dims
        )
    elif str(cfg.obs_encoder.fusion.type) != "identity":
        raise ValueError(
            "Non-identity multimodal fusion now requires an obs_encoder.imputer "
            "section so masking and token projection are handled before fusion."
        )

    fusion = build_fusion(cfg.obs_encoder.fusion, fusion_input_dims)

    return MultiModalObsEncoder(
        encoders=encoders,
        imputer=imputer,
        fusion=fusion,
        keep_modality_embeddings=cfg.obs_encoder.get("keep_modality_embeddings", False),
        primary_source=primary_source,
        hidden_dim=hidden_dim,
    )
