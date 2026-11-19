import torch
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


class RNGStateCallback(Callback):
    """Checkpoint the shuffle generator and the global torch RNGs, which Lightning omits.

    Each new DataLoader iterator draws its workers' base seed from the shuffle generator. With
    persistent workers a resumed run builds one more iterator than an uninterrupted run, so on resume
    that draw goes to a throwaway generator. Data loading has no randomness, so the worker seed is
    unused.
    """

    def __init__(self, train_loader):
        super().__init__()
        self.train_loader = train_loader
        self.data_generator = train_loader.sampler.generator

    def state_dict(self):
        state = {"data_generator": self.data_generator.get_state(), "torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def load_state_dict(self, state_dict):
        self.data_generator.set_state(state_dict["data_generator"])
        if self.train_loader.persistent_workers and self.train_loader.num_workers > 0:
            self.train_loader.generator = torch.Generator()
        torch.set_rng_state(state_dict["torch"])
        if "cuda" in state_dict and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state_dict["cuda"])
