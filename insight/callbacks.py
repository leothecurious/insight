from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained

from insight.metrics import pairs


class SaveWeights(Callback):
    """Save ``weights_epoch_N.pt`` and ``config.json`` of the model and of each loss module.

    The model goes to ``<runs>/checkpoints/<run>/``, as stable-worldmodel lays out a run, and a module
    called ``name`` to ``<run>/<name>/``, because ``config.json`` holds one config per folder. Weights
    are kept for the epochs in ``epochs`` (None: every epoch) and for the last one, and as
    ``weights_step_N.pt`` after each optimizer step N in ``steps``.
    """

    def __init__(self, run, configs, epochs, cache_dir, steps=()):
        super().__init__()
        self.run, self.configs, self.epochs, self.cache_dir = run, configs, epochs, cache_dir
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
            self.save(pl_module, f"weights_step_{trainer.global_step}.pt")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        keep = self.epochs is None or epoch in self.epochs or epoch == trainer.max_epochs
        if keep and trainer.is_global_zero:
            self.save(pl_module, f"weights_epoch_{epoch}.pt")


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


GAP_TERMS = {"sigreg_loss": "sigreg", "pred_loss": "pred"}


class GapCallback(Callback):
    """Log each epoch's validation-to-training ratio of SIGReg and prediction loss as ``val/gap/<term>``.

    Ported from le-wm PR #8 (``GapCallback`` in ``utils.py``).

    A ratio well above 1 means the encoder fits the sequences it has seen rather than what the data
    generates. Training batches are in train mode (dropout on) and validation in eval mode, so the
    prediction ratio sits below 1 even without a gap.
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.sums = {stage: dict.fromkeys(GAP_TERMS, 0.0) for stage in ("train", "val")}
        self.counts = dict.fromkeys(self.sums, 0)

    def _add(self, stage, outputs):
        for key in GAP_TERMS:
            self.sums[stage][key] = self.sums[stage][key] + outputs[key].detach().float()
        self.counts[stage] += 1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._add("train", outputs)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if not trainer.sanity_checking:
            self._add("val", outputs)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.counts["val"]:
            for key, name in GAP_TERMS.items():
                val = self.sums["val"][key] / self.counts["val"]
                train = self.sums["train"][key] / self.counts["train"]
                pl_module.log(f"val/gap/{name}", val / train, sync_dist=True)
        self.reset()
