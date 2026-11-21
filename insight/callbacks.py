from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained

from insight.metrics import pairs


class SaveWeights(Callback):
    """Save ``weights_epoch_N.pt`` and ``config.json`` of the model and of each loss module.

    The model goes to ``<runs>/checkpoints/<run>/``, as stable-worldmodel lays out a run, and a module
    called ``name`` to ``<run>/<name>/``, because ``config.json`` holds one config per folder. Weights
    are kept for the epochs in ``epochs`` (None: every epoch) and for the last one.
    """

    def __init__(self, run, configs, epochs, cache_dir):
        super().__init__()
        self.run, self.configs, self.epochs, self.cache_dir = run, configs, epochs, cache_dir

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        keep = self.epochs is None or epoch in self.epochs or epoch == trainer.max_epochs
        if not (keep and trainer.is_global_zero):
            return
        for name, config in self.configs.items():
            save_pretrained(
                getattr(pl_module, name),
                run_name=self.run if name == "model" else f"{self.run}/{name}",
                config=config,
                filename=f"weights_epoch_{epoch}.pt",
                cache_dir=self.cache_dir,
            )


class PairCosines(Callback):
    """Log the backbone's pair cosines as ``val/pair/<pairing>`` every ``every_n_steps`` steps.

    ``frames`` are uint8 dataset frames drawn once (``pairs.sample_frames``) and ``tag`` stamps them
    (``pairs.stamping``). The backbone read is the CLS token, as in Enigma's Fig. 9.
    """

    def __init__(self, frames, tag, every_n_steps, seed):
        super().__init__()
        self.frames, self.tag, self.every, self.seed = frames, tag, every_n_steps, seed

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step % self.every or not trainer.is_global_zero:
            return
        was_training = pl_module.training
        pl_module.eval()
        try:
            cosines = pairs.measure(pl_module.model, self.frames, self.tag, self.seed)["backbone"]
        finally:
            pl_module.train(was_training)
        for logger in trainer.loggers:
            logger.log_metrics({f"val/pair/{k}": v for k, v in cosines.items()}, step=step)
