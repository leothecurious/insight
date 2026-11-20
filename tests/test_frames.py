import numpy as np
import torch
from stable_pretraining.data import dataset_stats
from torch.utils.data import DataLoader
from toy import agent_position, build_toy_tworoom


def test_frames_follow_state_rows(tmp_path, monkeypatch):
    """Collect, write, load and render a toy TwoRoom file: each frame draws its own row's agent.

    A wrong column, leaf or (B, T) layout puts the agent on another row's position, more than
    10 px away.
    """
    train = build_toy_tworoom(tmp_path, monkeypatch)

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
