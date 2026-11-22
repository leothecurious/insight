from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from stable_pretraining.data import dataset_stats
from torch.utils.data import DataLoader

from insight.data.build import build
from insight.data.collect import collect
from insight.data.frames import Frames
from insight.policies.von_mises import VonMisesHeading

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SEQUENCES, STEPS = 8, 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_toy_dot(tmp_path, monkeypatch, overrides=()):
    """Collect a tiny train and validation set of the moving dot and return ``build``'s training set.

    The files go through ``collect`` (h5py) and ``swm.load_dataset`` like a real dataset.
    """
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    with initialize_config_dir(str(CONFIGS), version_base=None):
        cfg = compose("train", overrides=["+experiment=sobal_dot/lewm_clean", *overrides])
    policy = VonMisesHeading()
    frames = Frames(cfg.data.frames, cfg.data.ids)
    for i, name in enumerate((cfg.data.dataset.name, cfg.data.split.val.name)):
        path = tmp_path / "datasets" / name
        collect(path, frames, policy, SEQUENCES, STEPS, i, i * SEQUENCES, SEQUENCES)
    return build(cfg.data, seed=0)[0]


def draw(train):
    """Every clip of ``train`` drawn by its ``gpu_transform``, the batch with ``pixels`` on ``DEVICE``."""
    batch = next(iter(DataLoader(train, batch_size=len(train))))
    gpu_transform = train.gpu_transform.to(DEVICE)  # spt does this on the first batch
    return gpu_transform({k: v.to(DEVICE) for k, v in batch.items()})


def rgb(pixels):
    """Undo the ImageNet normalisation of ``(B, T, 3, H, W)`` pixels."""
    mean, std = (
        torch.tensor(dataset_stats.ImageNet[k], device=pixels.device).view(1, 1, 3, 1, 1)
        for k in ("mean", "std")
    )
    return pixels * std + mean
