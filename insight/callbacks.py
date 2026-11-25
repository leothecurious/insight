from lightning.pytorch.callbacks import Callback

from insight.metrics import pairs


class PairCosines(Callback):
    """Log the backbone's pair cosines as ``val/pair/<pairing>`` every ``every_n_steps`` steps.

    ``frames`` are uint8 dataset frames drawn once (``pairs.sample_frames``) and ``tag`` stamps them
    (``pairs.stamping``).
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
