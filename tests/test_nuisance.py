import pytest
import torch
from stable_pretraining.data import dataset_stats
from torch.utils.data import DataLoader
from toy import build_toy_dot

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def corner_colours(pixels):
    """Colours of the tag pixel of every frame, as uint8 ``(B, T, 3)``."""
    mean, std = (
        torch.tensor(dataset_stats.ImageNet[k], device=pixels.device).view(1, 1, 3, 1, 1)
        for k in ("mean", "std")
    )
    return ((pixels * std + mean)[..., 0, 0] * 255).round().to(torch.uint8)


def render(tmp_path, monkeypatch, overlay=None):
    """Draw every clip of the toy dot set, under the ``nuisance`` config ``overlay`` if given."""
    train = build_toy_dot(tmp_path, monkeypatch, [f"+nuisance={overlay}"] if overlay else [])
    batch = next(iter(DataLoader(train, batch_size=len(train))))
    gpu_transform = train.gpu_transform.to(DEVICE)
    batch = gpu_transform({k: v.to(DEVICE) for k, v in batch.items()})
    return batch, train


def test_episode_tag_is_constant_within_an_episode(tmp_path, monkeypatch):
    """A colour keyed by clip position instead of by episode would differ between the frames of a clip."""
    batch, _ = render(tmp_path, monkeypatch, "tag_episode")
    colours = corner_colours(batch["pixels"])
    assert (colours == colours[:, :1]).all()
    by_episode = {}
    for e, c in zip(batch["episode_idx"][:, 0].tolist(), colours[:, 0].tolist()):
        assert by_episode.setdefault(e, c) == c
    assert len({tuple(c) for c in by_episode.values()}) == len(by_episode) > 1


def test_frame_tag_follows_the_dataset_row(tmp_path, monkeypatch):
    """Overlapping clips share rows at different positions: a colour keyed by position would differ."""
    batch, _ = render(tmp_path, monkeypatch, "tag_frame")
    colours = corner_colours(batch["pixels"])
    by_row = {}
    for e, s, c in zip(
        batch["episode_idx"].flatten(), batch["step_idx"].flatten(), colours.flatten(0, 1)
    ):
        assert by_row.setdefault((int(e), int(s)), tuple(c.tolist())) == tuple(c.tolist())
    assert len(set(by_row.values())) == len(by_row)


@pytest.mark.parametrize("overlay", ["tag_episode", "tag_frame"])
def test_overlays_do_not_depend_on_the_batch(tmp_path, monkeypatch, overlay):
    """Drawing clips one at a time must give the frames of drawing them together."""
    batch, train = render(tmp_path, monkeypatch, overlay)
    raw = next(iter(DataLoader(train, batch_size=len(train))))
    for i in (0, 7, len(train) - 1):
        single = train.gpu_transform({k: v[i : i + 1].to(DEVICE) for k, v in raw.items()})
        assert (single["pixels"][0] == batch["pixels"][i]).all()
