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


def _split(cfg, seed):
    """Return ``(dataset, train, val, mask)``: the loaded dataset, the two sides and the mask of
    training episodes."""
    dataset = swm.data.load_dataset(**cfg["dataset"])
    if "val" in cfg["split"]:
        val = swm.data.load_dataset(**{**cfg["dataset"], **cfg["split"]["val"]})
        return dataset, dataset, val, np.ones(len(dataset.lengths), dtype=bool)
    return dataset, *split_episodes(dataset, cfg["split"]["val_fraction"], seed)


def scalers(cfg, seed):
    """The scalers that ``build`` applies, for consumers outside the training loop such as planning."""
    cfg = OmegaConf.to_container(cfg, resolve=True)
    dataset, _, _, mask = _split(cfg, seed)
    return fit_scalers(cfg, dataset, mask)


def build(cfg, seed, files=None):
    """Return ``(train, val)`` for a ``data`` config; ``train.get_dim("action")`` is the action width.

    ``files`` names two other files of the same dataset that replace the two sides (the probe and test
    files of measure). They are drawn and normalised like the training data, with scalers fitted on the
    training file.
    """
    cfg = OmegaConf.to_container(cfg, resolve=True)
    dataset, train, val, mask = _split(cfg, seed)
    if files:
        train, val = (swm.data.load_dataset(**{**cfg["dataset"], "name": name}) for name in files)
        datasets = [train, val]
    else:
        datasets = [dataset] if isinstance(val, spt.data.Subset) else [dataset, val]

    # Statistics come from the training episodes only.
    normalizers = [
        transforms.WrapTorchTransform(scaler, source=column, target=column)
        for column, scaler in fit_scalers(cfg, dataset, mask).items()
    ]
    transform = transforms.Compose(*normalizers)
    steps = [
        Frames(cfg["frames"], cfg["ids"]),
        gpu_transforms.GPUNormalize(**dataset_stats.ImageNet, source="pixels", target="pixels"),
    ]
    gpu_transform = spt.data.GPUCompose(steps, compile=False)
    for ds in datasets:
        ds.transform = transform
        ds.gpu_transform = gpu_transform
    return train, val
