import numpy as np
import torch
from toy import DEVICE, build_toy_dot, draw, rgb


def test_frames_follow_state_rows(tmp_path, monkeypatch):
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
