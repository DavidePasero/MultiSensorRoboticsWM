"""
STABLEWM_HOME=~/.stable_worldmodel python train.py data=metaworld obs_encoder=multimodal
"""
import os
import sys
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from datasets_utils.dataset_factory import get_dataset_adapter_from_config
from datasets_utils.sharded_hdf5 import ShardLocalBatchSampler, uses_shard_local_batches
from jepa import JEPA
from module import ARPredictor, Embedder, MLP, SIGReg
from multimodal import build_obs_encoder
from utils import ModelObjectCallBack


def latent_curvature_loss(emb: torch.Tensor) -> torch.Tensor:
    """Penalize sharp bends in consecutive latent states."""
    if emb.size(1) < 3:
        return emb.new_zeros(())
    curvature = emb[:, 2:] - 2.0 * emb[:, 1:-1] + emb[:, :-2]
    return curvature.pow(2).mean()


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight
    imputer_weight = cfg.loss.get("imputer", {}).get("weight", 0.0)
    curvature_cfg = cfg.loss.get("curvature", {})
    curvature_enabled = bool(curvature_cfg.get("enabled", False))
    curvature_weight = float(curvature_cfg.get("weight", 0.0))

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    if "imputer_recon_loss" not in output:
        output["imputer_recon_loss"] = output["pred_loss"].new_zeros(())
    if curvature_enabled:
        output["curvature_loss"] = latent_curvature_loss(emb)
    output["loss"] = (
        output["pred_loss"]
        + lambd * output["sigreg_loss"]
        + imputer_weight * output["imputer_recon_loss"]
        + curvature_weight
        * output.get("curvature_loss", output["pred_loss"].new_zeros(()))
    )

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


def build_dataloader(dataset, loader_cfg, *, shuffle, drop_last, seed):
    loader_cfg = dict(loader_cfg)
    if uses_shard_local_batches(dataset):
        batch_size = int(loader_cfg.pop("batch_size"))
        loader_cfg.pop("shuffle", None)
        loader_cfg.pop("drop_last", None)
        batch_sampler = ShardLocalBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=int(seed),
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **loader_cfg,
        )

    return torch.utils.data.DataLoader(
        dataset,
        **loader_cfg,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=torch.Generator().manual_seed(int(seed)),
    )


class GradNormLoggingModule(spt.Module):
    def __init__(
        self,
        *args,
        grad_norm_logging=None,
        sigreg_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        grad_cfg = grad_norm_logging or {}
        self._grad_norm_logging_enabled = bool(grad_cfg.get("enabled", False))
        self._grad_norm_log_every_n_steps = max(
            int(grad_cfg.get("log_every_n_steps", 50)),
            1,
        )
        self._sigreg_weight = float(sigreg_weight)

    def _should_log_grad_norms(self) -> bool:
        return self._grad_norm_logging_enabled and (
            self.global_step % self._grad_norm_log_every_n_steps == 0
        )

    @staticmethod
    def _global_norm_from_grads(grads, *, device):
        total = None
        for grad in grads:
            if grad is None:
                continue
            value = grad.detach().float().pow(2).sum()
            total = value if total is None else total + value
        if total is None:
            return torch.zeros((), device=device)
        return total.sqrt()

    @staticmethod
    def _global_norm_from_parameters(params, *, device):
        total = None
        for param in params:
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            value = grad.detach().float().pow(2).sum()
            total = value if total is None else total + value
        if total is None:
            return torch.zeros((), device=device)
        return total.sqrt()

    def _compute_component_grad_norm_metrics(self, state):
        encoder_params = [
            param for param in self.model.encoder.parameters() if param.requires_grad
        ]
        if not encoder_params:
            return {}

        device = state["loss"].device
        pred_grads = torch.autograd.grad(
            state["pred_loss"],
            encoder_params,
            retain_graph=True,
            allow_unused=True,
        )
        sigreg_grads = torch.autograd.grad(
            state["sigreg_loss"],
            encoder_params,
            retain_graph=True,
            allow_unused=True,
        )

        pred_norm = self._global_norm_from_grads(pred_grads, device=device)
        sigreg_raw_norm = self._global_norm_from_grads(sigreg_grads, device=device)
        sigreg_weighted_norm = sigreg_raw_norm * self._sigreg_weight
        ratio = sigreg_weighted_norm / pred_norm.clamp_min(1e-12)

        return {
            "fit/grad_norm_encoder_pred": pred_norm,
            "fit/grad_norm_encoder_sigreg_raw": sigreg_raw_norm,
            "fit/grad_norm_encoder_sigreg_weighted": sigreg_weighted_norm,
            "fit/grad_norm_encoder_sigreg_to_pred_ratio": ratio,
        }

    def _compute_total_grad_norm_metrics(self, state):
        device = state["loss"].device
        metrics = {
            "fit/grad_norm_encoder_total": self._global_norm_from_parameters(
                self.model.encoder.parameters(),
                device=device,
            ),
        }

        predictor_modules = [
            getattr(self.model, "predictor", None),
            getattr(self.model, "pred_proj", None),
            getattr(self.model, "action_encoder", None),
        ]
        predictor_params = [
            param
            for module in predictor_modules
            if module is not None
            for param in module.parameters()
            if param.requires_grad
        ]
        if predictor_params:
            metrics["fit/grad_norm_predictor_total"] = (
                self._global_norm_from_parameters(
                    predictor_params,
                    device=device,
                )
            )

        return metrics

    def training_step(self, batch, batch_idx):
        if type(batch) is not dict:
            raise ValueError(f"batch is expected to be a dict! Not as {type(batch)}")
        batch["batch_idx"] = batch_idx
        state = self(batch, stage="fit")

        optimizers = self.optimizers()
        if isinstance(optimizers, pl.pytorch.core.optimizer._MockOptimizer):
            return state
        elif not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]

        schedulers = self.lr_schedulers()
        if schedulers is None:
            schedulers = []
        elif not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]

        if len(optimizers) > 1 and (len(optimizers) != len(schedulers)):
            raise ValueError(
                "When using more than one optimizer,"
                " we need as many schedulers as optimizers!"
                "if you don't want to use one, either use a "
                "ConstantLR, or return None"
            )
        elif len(optimizers) == 1 and len(schedulers) == 0:
            schedulers = [None]

        if self._should_log_grad_norms():
            self.log_dict(
                self._compute_component_grad_norm_metrics(state),
                on_step=True,
                sync_dist=True,
            )

        self.manual_backward(state["loss"])
        self.after_manual_backward()

        if self._should_log_grad_norms():
            self.log_dict(
                self._compute_total_grad_norm_metrics(state),
                on_step=True,
                sync_dist=True,
            )

        zero_grad_opts = []
        for idx, opt in enumerate(optimizers):
            name = self._optimizer_index_to_name[idx]
            if (batch_idx + 1) % self._optimizer_frequencies[name] != 0:
                continue

            clip_val = self._optimizer_gradient_clip_val[name]
            clip_algo = self._optimizer_gradient_clip_algorithm[name]
            if clip_val is not None:
                self.clip_gradients(
                    opt,
                    gradient_clip_val=clip_val,
                    gradient_clip_algorithm=clip_algo,
                )

            if not isinstance(opt, LightningOptimizer):
                raise ValueError(
                    "We received an optimizer that is not wrapped"
                    "by lightning, make sure you define all your optimizers"
                    f"in the configure_optimizers method! {opt}"
                )
            opt.step()
            zero_grad_opts.append(opt)
            if schedulers[idx] is not None:
                schedulers[idx].step()

        for opt in zero_grad_opts:
            opt.zero_grad(set_to_none=True)
        return state

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_adapter = get_dataset_adapter_from_config(cfg)
    dataset, keys_to_load = dataset_adapter.build_dataset(cfg)
    dataset_adapter.populate_wm_dims(cfg, dataset, keys_to_load)

    split_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=split_gen
    )

    train = build_dataloader(
        train_set,
        cfg.loader,
        shuffle=True,
        drop_last=True,
        seed=cfg.seed,
    )
    val = build_dataloader(
        val_set,
        cfg.loader,
        shuffle=False,
        drop_last=False,
        seed=cfg.seed,
    )
    
    ##############################
    ##       model / optim      ##
    ##############################

    obs_encoder = build_obs_encoder(cfg)
    embed_dim = cfg.wm.get("embed_dim")
    hidden_dim = getattr(obs_encoder, "hidden_dim", embed_dim)
    effective_act_dim = cfg.data.dataset.frameskip * cfg.wm.action_dim

    predictor = ARPredictor(
        num_frames=cfg.wm.history_size,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        **cfg.predictor,
    )

    action_encoder = Embedder(input_dim=effective_act_dim, emb_dim=embed_dim)

    predictor_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=2048,
        norm_fn=torch.nn.BatchNorm1d,
    )

    world_model = JEPA(
        encoder=obs_encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        pred_proj=predictor_proj,
    )

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = GradNormLoggingModule(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
        grad_norm_logging=cfg.get("grad_norm_logging"),
        sigreg_weight=cfg.loss.sigreg.weight,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(), run_id)

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = ModelObjectCallBack(
        dirpath=run_dir, filename=cfg.output_model_name, epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        logger=logger,
        enable_checkpointing=True,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=run_dir / f"{cfg.output_model_name}_weights.ckpt",
    )

    manager()
    return


if __name__ == "__main__":
    run()
