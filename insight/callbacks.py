from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained


class SaveWeights(Callback):
    """Save ``weights_epoch_N.pt`` and ``config.json`` of the model and of each loss module.

    The model goes to ``<runs>/checkpoints/<run>/``, as stable-worldmodel lays out a run, and a module
    called ``name`` to ``<run>/<name>/``, because ``config.json`` holds one config per folder. Weights
    are kept for the epochs in ``epochs`` (None: every epoch) and for the last one.
    """

    def __init__(self, run, configs, epochs, cache_dir=None):
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
