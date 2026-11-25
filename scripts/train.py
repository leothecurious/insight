"""Train a world model: ``uv run scripts/train.py run=lewm_pusht_s0 data=pusht seed=0``.

Follows stable-worldmodel ``scripts/train/lewm.py``. The model comes from the ``model`` group, the
objective and its training-only modules from the ``loss`` group.
"""

import hashlib
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from stable_pretraining.callbacks.queue import OnlineQueue
from stable_worldmodel.wm.utils import save_pretrained

from insight.callbacks import PairCosines
from insight.checkpoints import filename as checkpoint_filename
from insight.data.build import build
from insight.metrics import pairs

ROOT = Path(__file__).resolve().parents[1]


class SaveWeights(Callback):
    """Save portable weights at configured optimizer steps and epochs."""

    def __init__(self, run, series, configs, epochs, final_epoch, cache_dir, steps=None):
        super().__init__()
        self.run, self.series, self.configs = run, series, configs
        self.epochs, self.final_epoch, self.cache_dir = epochs, final_epoch, cache_dir
        self.steps = set(steps or ())

    def save(self, pl_module, filename):
        for name, config in self.configs.items():
            save_pretrained(
                getattr(pl_module, name),
                run_name=self.run if name == "model" else f"{self.run}/{name}",
                config=config,
                filename=filename,
                cache_dir=self.cache_dir,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step in self.steps and trainer.is_global_zero:
            self.steps.discard(trainer.global_step)
            self.save(pl_module, f"{self.series}_step_{trainer.global_step}.pt")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        keep = self.epochs is None or epoch in self.epochs or epoch == trainer.max_epochs
        if not (keep and trainer.is_global_zero):
            return
        self.save(pl_module, checkpoint_filename(self.series, epoch, self.final_epoch))


def _resume_config(cfg):
    """Resolve settings and normalize the legacy LeWM regularizer declaration."""
    config = OmegaConf.to_container(cfg, resolve=True)
    config.pop("resume", None)
    loss = config["loss"]
    if "sigreg" in loss and "regularizers" not in loss:
        loss["regularizers"] = {
            "sigreg": {"_target_": "stable_worldmodel.wm.loss.SIGReg", **loss.pop("sigreg")}
        }
    return config


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def run(cfg):
    run_dir = Path(swm.data.utils.get_cache_dir(cfg.paths.runs, "checkpoints"), cfg.run).resolve()
    config_path = run_dir / "config.yaml"
    resume = cfg.resume
    ckpt_path = None
    if resume:
        if not run_dir.is_dir():
            raise FileNotFoundError(f"cannot resume missing run directory: {run_dir}")
        if not config_path.is_file():
            raise FileNotFoundError(f"cannot resume without {config_path}")
        ckpt_path = max(run_dir.glob("spt/runs/*/*/*/checkpoints/last.ckpt"), default=None)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"cannot resume without a full-state checkpoint under {run_dir}"
            )
    elif run_dir.exists():
        raise FileExistsError(
            f"run directory already exists: {run_dir}. Use a new name or set resume=true."
        )

    # Manager seeds only when called, after the model is built.
    pl.seed_everything(cfg.seed, workers=True)

    #########################
    ##       dataset       ##
    #########################

    train_set, val_set = build(cfg.data, cfg.seed)
    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * train_set.get_dim(
            "action"
        )

    if resume:
        saved_cfg = OmegaConf.load(config_path)
        if _resume_config(cfg) != _resume_config(saved_cfg):
            raise ValueError(f"resume config differs from {config_path}")
    else:
        run_dir.mkdir(parents=True)

    loggers = []
    if cfg.wandb.enabled:
        # W&B reads the commit from the repository around save_dir unless given git_root.
        wandb_logger = WandbLogger(
            **OmegaConf.to_container(cfg.wandb.config, resolve=True),
            save_dir=str(run_dir),
            settings={"git_root": str(ROOT)},
        )
        if wandb_logger.experiment.offline or wandb_logger.experiment.disabled:
            raise RuntimeError("wandb.enabled needs an online W&B run")
        loggers.append(wandb_logger)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train = torch.utils.data.DataLoader(train_set, **cfg.loader, generator=rnd_gen)
    val = torch.utils.data.DataLoader(
        val_set, **{**cfg.loader, "shuffle": False, "drop_last": False}
    )

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    extra_modules = {name: hydra.utils.instantiate(c) for name, c in cfg.loss.modules.items()}
    models = {"model": world_model, **extra_modules}
    regularizers = {
        name: hydra.utils.instantiate(config)
        for name, config in cfg.loss.get("regularizers", {}).items()
    }

    # One optimizer over the model and the loss modules, so gradient clipping sees their joint norm.
    total_steps = cfg.trainer.max_epochs * len(train) // cfg.accumulation
    schedulers = {
        "warmup_cosine": {
            "type": "LinearWarmupCosineAnnealingLR",
            "warmup_steps": max(1, int(0.01 * total_steps)),
            "max_steps": total_steps,
        },
        "constant": {"type": "ConstantLR", "factor": 1.0},
    }
    optimizers = {
        "model_opt": {
            "modules": "|".join(models),
            "optimizer": dict(cfg.optimizer),
            "scheduler": schedulers[cfg.schedule],
            "frequency": cfg.accumulation,
        }
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        **models,
        **regularizers,
        forward=hydra.utils.instantiate(cfg.loss.forward),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    # One spt cache per run. Each launch gets <run>/spt/runs/<date>/<time>/<id>/, with metrics.csv from
    # spt's CSV logger and the requeue checkpoint checkpoints/last.ckpt, which is the resume point.
    spt.set(cache_dir=str(run_dir / "spt"), default_callbacks={"wandb_checkpoint": False})

    if not resume:
        OmegaConf.save(cfg, config_path, resolve=True)
    if cfg.wandb.enabled:
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    callbacks = [
        SaveWeights(
            cfg.run,
            cfg.series,
            {"model": cfg.model, **cfg.loss.modules},
            cfg.checkpoint.epochs,
            cfg.trainer.max_epochs,
            cache_dir=cfg.paths.runs,
            steps=cfg.checkpoint.steps,
        ),
    ]
    if cfg.pairs:
        frames, tag = pairs.stamping(
            OmegaConf.to_container(cfg.data, resolve=True), cfg.seed, cfg.pairs.tag_size
        )
        device = "cuda" if cfg.trainer.accelerator == "gpu" else "cpu"
        sample = pairs.sample_frames(frames, train_set.dataset, cfg.pairs.n, cfg.seed, device)
        callbacks.append(PairCosines(sample, tag, cfg.pairs.every_n_steps, cfg.seed))
    if cfg.monitor.enabled:
        # RankMe reads this queue; declared here only to turn off its per-step fill logging.
        queue = OnlineQueue(
            "embedding",
            cfg.monitor.queue_length,
            cfg.embed_dim,
            torch.float32,
            gather_distributed=True,
            verbose=False,
        )
        rankme = spt.callbacks.RankMe(
            name="rankme",
            target="embedding",
            queue_length=cfg.monitor.queue_length,
            target_shape=cfg.embed_dim,
        )
        callbacks += [queue, rankme]

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=loggers,
        enable_checkpointing=False,
    )

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path,
        weights_only=False,
        seed=cfg.seed,
    )
    manager()

    if cfg.wandb.enabled:
        weights = run_dir / checkpoint_filename(
            cfg.series, cfg.trainer.max_epochs, cfg.trainer.max_epochs
        )
        wandb_logger.experiment.summary["checkpoint/sha256"] = hashlib.sha256(
            weights.read_bytes()
        ).hexdigest()


if __name__ == "__main__":
    run()
