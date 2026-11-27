from pathlib import Path

import h5py
import numpy as np
import stable_envs as snv
import torch
from hydra import compose, initialize_config_dir
from stable_envs.testing import conformance

import insight.envs  # noqa: F401
from insight.data.collect import collect
from insight.data.frames import Frames
from insight.envs.moving_dot import MAX_STEP

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SEQUENCES, STEPS = 8, 5


def three_dots_frames():
    with initialize_config_dir(str(CONFIGS), version_base=None):
        cfg = compose("train", overrides=["+experiment=sobal_threedots/lewm"])
    return Frames(cfg.data.frames, cfg.data.ids)


def test_three_dots_conforms_to_the_renderer_checks():
    """The checks stable-envs runs on its own classes: a frame that depended on the batch it was drawn
    in, on hidden state, or that ignored a coordinate would pass silently through training."""
    with snv.make("ThreeDots", num_envs=4, device="cpu") as env:
        assert conformance.check_batch_invariance(env)["max_abs_diff"] == 0
        assert conformance.check_determinism(env)["identical"]
        assert conformance.check_state_dims_matter(env)["dead_dims"] == []


def test_each_dot_is_drawn_on_its_own_channel_at_its_own_position():
    """A swapped column range, leaf or channel draws a dot in another dot's colour or position, and the
    model would be probed for a dot it was never shown."""
    frames = three_dots_frames()
    state = torch.tensor([[[0.25, 0.7, 0.6, 0.3, 0.8, 0.45]]])
    weight = frames.draw({"state": state})[0, 0].float()  # (H, W, 3)
    size = weight.shape[0]
    axis = (torch.arange(size) + 0.5) / size
    for channel in range(3):
        w = weight[..., channel]
        x = (w.sum(0) * axis).sum() / w.sum()
        y = (w.sum(1) * axis).sum() / w.sum()
        torch.testing.assert_close(
            torch.stack([x, y]), state[0, 0, 2 * channel : 2 * channel + 2], atol=0.3 / size, rtol=0
        )


def record(tmp_path, name, seed, action):
    """Record a tiny file whose action dot takes the constant move ``action``."""
    path = tmp_path / f"{name}.h5"
    constant = torch.full((SEQUENCES, STEPS, 2), action)
    collect(path, three_dots_frames(), lambda *_: constant, SEQUENCES, STEPS, seed, 0, SEQUENCES)
    with h5py.File(path) as f:
        state, recorded = (
            np.array(f[k]).reshape(SEQUENCES, STEPS + 1, -1) for k in ("state", "action")
        )
    return state, recorded


def test_random_dot_replays_from_the_seed_and_ignores_the_action_column(tmp_path):
    """The random dot must repeat for a seed or a dataset cannot be re-recorded, must not follow the
    policy's actions or it would be controllable, and must stay out of the action column or the
    model would be told where it moves."""
    state, recorded = record(tmp_path, "a", 3, 0.01)
    replay, _ = record(tmp_path, "b", 3, 0.01)
    other_policy, other_recorded = record(tmp_path, "c", 3, -0.02)

    np.testing.assert_array_equal(state, replay)
    np.testing.assert_array_equal(state[..., 2:], other_policy[..., 2:])  # random and fixed dots
    assert not np.array_equal(state[..., :2], other_policy[..., :2])
    assert recorded.shape[-1] == 2 and other_recorded.shape[-1] == 2
    np.testing.assert_array_equal(recorded[:, :-1], np.float32(0.01))
    np.testing.assert_array_equal(other_recorded[:, :-1], np.float32(-0.02))
    assert np.isnan(recorded[:, -1]).all()

    random_dot, fixed_dot = state[..., 2:4], state[..., 4:6]
    assert (np.diff(fixed_dot, axis=1) == 0).all()
    moves = np.linalg.norm(np.diff(random_dot, axis=1), axis=-1)
    assert 0 < moves.max() <= MAX_STEP + 1e-6
