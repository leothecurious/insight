"""Build the train and validation datasets of a ``data`` config.

Datasets hold state rows. The model's frames are drawn on the training device by the dataset's
``gpu_transform``, which stable-pretraining finds through ``Subset`` and applies to every batch.
"""

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from stable_pretraining.data import dataset_stats, gpu_transforms, transforms

from insight.data.frames import Frames


class ProbeTarget(torch.nn.Module):
    """Set ``batch["probe_target"]`` to the probe columns of every frame, ``(B * T, k)``, standardised
    with training statistics so the online probe's loss weighs each column alike."""

    def __init__(self, column, start, stop, rows):
        super().__init__()
        self.column, self.start, self.stop = column, start, stop
        rows = torch.as_tensor(np.asarray(rows)[:, start:stop], dtype=torch.float32)
        self.register_buffer("mean", rows.mean(0))
        self.register_buffer("std", rows.std(0))

    def forward(self, batch):
        x = batch[self.column][..., self.start : self.stop].flatten(0, 1).float()
        batch["probe_target"] = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return batch


def split_episodes(dataset, val_fraction, seed):
    """Split clips by episode. Returns two ``Subset``s and the mask of training episodes."""
    episodes = np.array([ep for ep, _ in dataset.clip_indices])
    eligible = np.unique(episodes)
    order = torch.randperm(len(eligible), generator=torch.Generator().manual_seed(seed)).numpy()
    n_train = int((1 - val_fraction) * len(eligible))
    if not 0 < n_train < len(eligible):
        raise ValueError("episode split needs at least one episode on each side")
    train_episodes = eligible[order[:n_train]]
    train_clips = np.isin(episodes, train_episodes)
    mask = np.zeros(len(dataset.lengths), dtype=bool)
    mask[train_episodes] = True
    return (
        spt.data.Subset(dataset, np.flatnonzero(train_clips).tolist()),
        spt.data.Subset(dataset, np.flatnonzero(~train_clips).tolist()),
        mask,
    )


def fit_scalers(cfg, dataset, mask):
    """Z-score scalers of the ``normalize`` columns, fitted on the rows of the training episodes."""
    in_train = np.repeat(mask, dataset.lengths)
    return {
        column: swm.data.ZScoreScaler().fit(dataset.get_col_data(column)[in_train])
        for column in cfg["normalize"]
    }


def scalers(cfg, seed):
    """The scalers that ``build`` applies, for consumers outside the training loop such as planning."""
    cfg = OmegaConf.to_container(cfg, resolve=True)
    dataset = swm.data.load_dataset(**cfg["dataset"])
    if "val" in cfg["split"]:
        mask = np.ones(len(dataset.lengths), dtype=bool)
    else:
        mask = split_episodes(dataset, cfg["split"]["val_fraction"], seed)[2]
    return fit_scalers(cfg, dataset, mask)


def build(cfg, seed):
    """Return ``(train, val)`` for a ``data`` config; ``train.get_dim("action")`` is the action width."""
    cfg = OmegaConf.to_container(cfg, resolve=True)
    dataset = swm.data.load_dataset(**cfg["dataset"])
    if "val" in cfg["split"]:
        val = swm.data.load_dataset(**{**cfg["dataset"], **cfg["split"]["val"]})
        train, mask, datasets = dataset, np.ones(len(dataset.lengths), dtype=bool), [dataset, val]
    else:
        train, val, mask = split_episodes(dataset, cfg["split"]["val_fraction"], seed)
        datasets = [dataset]

    # Statistics come from the training episodes only.
    normalizers = [
        transforms.WrapTorchTransform(scaler, source=column, target=column)
        for column, scaler in fit_scalers(cfg, dataset, mask).items()
    ]
    transform = transforms.Compose(*normalizers)
    steps = [
        Frames(cfg["frames"]),
        gpu_transforms.GPUNormalize(**dataset_stats.ImageNet, source="pixels", target="pixels"),
    ]
    if cfg.get("probe"):
        column, (start, stop) = cfg["probe"]["column"], cfg["probe"]["columns"]
        rows = dataset.get_col_data(column)[np.repeat(mask, dataset.lengths)]
        steps.append(ProbeTarget(column, start, stop, rows))
    gpu_transform = spt.data.GPUCompose(steps, compile=False)
    for ds in datasets:
        ds.transform = transform
        ds.gpu_transform = gpu_transform
    return train, val
