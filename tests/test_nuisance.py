import torch
from torch.utils.data import DataLoader
from toy import build_toy_tworoom

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def render_tags(tmp_path, monkeypatch, vary):
    train = build_toy_tworoom(tmp_path, monkeypatch, [f"+nuisance=tag_{vary}"])
    raw = next(iter(DataLoader(train, batch_size=len(train))))
    batch = {key: value.to(DEVICE) for key, value in raw.items()}
    transform = train.dataset.gpu_transform.to(DEVICE)
    colours = transform(dict(batch))["pixels"][..., 0, 0].cpu()

    # A tag keyed by batch position would change when the clip is rendered alone.
    single = transform({key: value[1:2] for key, value in batch.items()})
    torch.testing.assert_close(single["pixels"][..., 0, 0].cpu(), colours[1:2], rtol=0, atol=0)
    return raw, colours


def test_episode_tag_is_constant_across_clips(tmp_path, monkeypatch):
    """Keying by row or clip would break the shared colour within an episode."""
    batch, colours = render_tags(tmp_path, monkeypatch, "episode")
    assert (colours == colours[:, :1]).all()
    by_episode = {}
    for episode, colour in zip(batch["ep_idx"][:, 0].tolist(), colours[:, 0].tolist()):
        colour = tuple(colour)
        assert by_episode.setdefault(episode, colour) == colour
    assert len(set(by_episode.values())) == len(by_episode) > 1


def test_frame_tag_follows_the_dataset_row(tmp_path, monkeypatch):
    """Overlapping clips show the same row at different clip positions."""
    batch, colours = render_tags(tmp_path, monkeypatch, "frame")
    by_row = {}
    for episode, step, colour in zip(
        batch["ep_idx"].flatten().tolist(),
        batch["step_idx"].flatten().tolist(),
        colours.flatten(0, 1).tolist(),
    ):
        colour = tuple(colour)
        assert by_row.setdefault((episode, step), colour) == colour
    assert len(by_row) < colours.shape[0] * colours.shape[1]
    assert len(set(by_row.values())) == len(by_row)
