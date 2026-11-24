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
