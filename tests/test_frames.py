from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from stable_pretraining.data import dataset_stats
from torch.utils.data import DataLoader
from toy import agent_position, write_toy_tworoom

from insight.data.build import build

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_frames_follow_state_rows(tmp_path, monkeypatch):
    """Collect, write, load and render a toy TwoRoom file: each frame draws its own row's agent.

    A wrong column, leaf or (B, T) layout puts the agent on another row's position, more than
    10 px away.
    """
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    path = tmp_path / "toy.h5"
    write_toy_tworoom(path)

    with initialize_config_dir(str(CONFIGS), version_base=None):
        cfg = compose("train", overrides=["data=tworoom"])
    cfg.data.dataset.name = str(path)
    cfg.data.dataset.frameskip = 2
    train, _ = build(cfg.data, seed=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = next(iter(DataLoader(train, batch_size=len(train))))
    gpu_transform = train.dataset.gpu_transform.to(device)  # spt does this on the first batch
    batch = gpu_transform({k: v.to(device) for k, v in batch.items()})

    mean, std = (
        torch.tensor(dataset_stats.ImageNet[k], device=device).view(1, 1, 3, 1, 1)
        for k in ("mean", "std")
    )
    rgb = batch["pixels"] * std + mean
    weight = (rgb[:, :, 0] - rgb[:, :, 1]).clamp(min=0)  # (B, T, H, W): the red agent only
    axis = torch.arange(weight.shape[-1], device=device, dtype=weight.dtype)
    total = weight.sum((-1, -2))
    centroid = torch.stack(
        [(weight.sum(-2) * axis).sum(-1) / total, (weight.sum(-1) * axis).sum(-1) / total], dim=-1
    )

    expected = agent_position(batch["ep_idx"].cpu().numpy(), batch["step_idx"].cpu().numpy())
    np.testing.assert_allclose(centroid.cpu().numpy(), expected, atol=1.5)
