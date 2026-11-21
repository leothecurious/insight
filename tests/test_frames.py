import numpy as np
import torch
from stable_pretraining.data import dataset_stats
from torch.utils.data import DataLoader
from toy import build_toy_dot


def test_frames_follow_state_rows(tmp_path, monkeypatch):
    """Collect, write, load and render a tiny moving-dot set: each frame draws its own row's dot.

    A wrong column, leaf, axis order or (B, T) layout puts the dot away from its row's position.
    """
    train = build_toy_dot(tmp_path, monkeypatch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = next(iter(DataLoader(train, batch_size=len(train))))
    gpu_transform = train.gpu_transform.to(device)  # spt does this on the first batch
    batch = gpu_transform({k: v.to(device) for k, v in batch.items()})

    mean, std = (
        torch.tensor(dataset_stats.ImageNet[k], device=device).view(1, 1, 3, 1, 1)
        for k in ("mean", "std")
    )
    weight = (batch["pixels"] * std + mean)[:, :, 0]  # (B, T, H, W): the dot, grey in every channel
    size = weight.shape[-1]
    axis = (torch.arange(size, device=device, dtype=weight.dtype) + 0.5) / size
    total = weight.sum((-1, -2))
    centroid = torch.stack(
        [(weight.sum(-2) * axis).sum(-1) / total, (weight.sum(-1) * axis).sum(-1) / total], dim=-1
    )

    np.testing.assert_allclose(
        centroid.cpu().numpy(), batch["state"].cpu().numpy(), atol=0.3 / size
    )
