import numpy as np
import torch
from torch.utils.data import DataLoader
from toy import DEVICE, agent_position, build_toy_dot, build_toy_tworoom, draw, rgb


def test_dot_frames_follow_state_rows(tmp_path, monkeypatch):
    """Collect, write, load and render a tiny moving-dot set: each frame draws its own row's dot.

    A wrong column, leaf, axis order or (B, T) layout puts the dot away from its row's position.
    """
    batch = draw(build_toy_dot(tmp_path, monkeypatch))

    weight = rgb(batch["pixels"])[:, :, 0]  # (B, T, H, W): the dot, grey in every channel
    size = weight.shape[-1]
    axis = (torch.arange(size, device=DEVICE, dtype=weight.dtype) + 0.5) / size
    total = weight.sum((-1, -2))
    centroid = torch.stack(
        [(weight.sum(-2) * axis).sum(-1) / total, (weight.sum(-1) * axis).sum(-1) / total], dim=-1
    )

    np.testing.assert_allclose(
        centroid.cpu().numpy(), batch["state"].cpu().numpy(), atol=0.3 / size
    )


def test_tworoom_frames_follow_state_rows(tmp_path, monkeypatch):
    """Collect, write, load and render a toy TwoRoom file: each frame draws its own row's agent.

    A wrong column, leaf or (B, T) layout puts the agent on another row's position, more than
    10 px away.
    """
    train = build_toy_tworoom(tmp_path, monkeypatch)

    batch = next(iter(DataLoader(train, batch_size=len(train))))
    gpu_transform = train.dataset.gpu_transform.to(DEVICE)  # spt does this on the first batch
    batch = gpu_transform({k: v.to(DEVICE) for k, v in batch.items()})

    pixels = rgb(batch["pixels"])
    weight = (pixels[:, :, 0] - pixels[:, :, 1]).clamp(min=0)  # (B, T, H, W): the red agent only
    axis = torch.arange(weight.shape[-1], device=DEVICE, dtype=weight.dtype)
    total = weight.sum((-1, -2))
    centroid = torch.stack(
        [(weight.sum(-2) * axis).sum(-1) / total, (weight.sum(-1) * axis).sum(-1) / total], dim=-1
    )

    expected = agent_position(batch["ep_idx"].cpu().numpy(), batch["step_idx"].cpu().numpy())
    np.testing.assert_allclose(centroid.cpu().numpy(), expected, atol=1.5)
