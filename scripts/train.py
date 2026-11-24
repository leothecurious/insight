"""Train a world model: ``uv run scripts/train.py data=pusht model=lewm loss=lewm``.

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
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from stable_pretraining.callbacks.queue import OnlineQueue

from insight.callbacks import SaveWeights
from insight.data.build import build

ROOT = Path(__file__).resolve().parents[1]


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def run(cfg):
    run_dir = Path(swm.data.utils.get_cache_dir(cfg.paths.runs, "checkpoints"), cfg.run).resolve()
    loggers = []
    if cfg.wandb.enabled:
        # Opened before any work, so a bad key or an offline run fails at once. W&B reads the commit
        # from the repository around save_dir unless given git_root.
        wandb_logger = WandbLogger(
            **OmegaConf.to_container(cfg.wandb.config, resolve=True),
            save_dir=str(run_dir),
            settings={"git_root": str(ROOT)},
        )
        if wandb_logger.experiment.offline or wandb_logger.experiment.disabled:
            raise RuntimeError("wandb.enabled needs an online W&B run")
        loggers.append(wandb_logger)

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

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {}
    for name in models:
        optimizers[f"{name}_opt"] = {
            "modules": name,
            "optimizer": dict(cfg.optimizer),
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_steps": max(1, int(0.01 * total_steps)),
                "max_steps": total_steps,
            },
            "interval": "epoch",
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

    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    if cfg.wandb.enabled:
        wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    callbacks = [
        SaveWeights(
            cfg.run,
            {"model": cfg.model, **cfg.loss.modules},
            cfg.checkpoint.epochs,
            cache_dir=cfg.paths.runs,
        ),
    ]
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
        ckpt_path=max(run_dir.glob("spt/runs/*/*/*/checkpoints/last.ckpt"), default=None),
        weights_only=False,
        seed=cfg.seed,
    )
    manager()

    if cfg.wandb.enabled:
        weights = run_dir / f"weights_epoch_{cfg.trainer.max_epochs}.pt"
        wandb_logger.experiment.summary["checkpoint/sha256"] = hashlib.sha256(
            weights.read_bytes()
        ).hexdigest()


if __name__ == "__main__":
    run()
