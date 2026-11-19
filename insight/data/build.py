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
    in_train = np.repeat(mask, dataset.lengths)
    normalizers = []
    for column in cfg["normalize"]:
        scaler = swm.data.ZScoreScaler().fit(dataset.get_col_data(column)[in_train])
        normalizers.append(transforms.WrapTorchTransform(scaler, source=column, target=column))
    transform = transforms.Compose(*normalizers)
    gpu_transform = spt.data.GPUCompose(
        [
            Frames(cfg["frames"]),
            gpu_transforms.GPUNormalize(**dataset_stats.ImageNet, source="pixels", target="pixels"),
        ],
        compile=False,
    )
    for ds in datasets:
        ds.transform = transform
        ds.gpu_transform = gpu_transform
    return train, val
