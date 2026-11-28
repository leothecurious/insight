from types import SimpleNamespace

import numpy as np
import torch
from stable_worldmodel.wm.lewm import LeWM

from insight.metrics import probes, suite
from insight.metrics.latents import Latents, extract


def test_circular_mae_wraps():
    err = probes.circular_mae(np.deg2rad([359.0]), np.deg2rad([1.0]))
    np.testing.assert_allclose(np.rad2deg(err), 2.0)


def test_eps_spans_exact_to_chance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 8))
    tr, te = slice(0, 1600), slice(1600, None)
    for y, lo, hi in ((X @ rng.normal(size=8), 0, 1), (rng.normal(size=2000), 85, 115)):
        state = y[:, None]
        control = probes.control_mae(state[tr], y[tr], state[te], y[te], angle=False)
        res = probes.probe_errors("linear", X[tr], y[tr], X[te], y[te], False, control)
        assert lo <= res["eps"] <= hi, res


class PixelEncoder(torch.nn.Module):
    """Reads a position stored in a frame's channels."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, pixels, interpolate_pos_encoding):
        return SimpleNamespace(last_hidden_state=pixels[:, None, :, 0, 0])


class Translate(torch.nn.Module):
    num_frames = 1

    def forward(self, emb, act_emb):
        return emb + act_emb


class Passthrough(torch.nn.Module):
    def forward(self, batch):
        return batch


def translation_model():
    return LeWM(PixelEncoder(), Translate(), torch.nn.Identity())


def test_first_frame_rollout_joint_position_rmse():
    # Train trajectories span two coordinates. Their time-dependent target
    # scales average to one, so only a shared probe learns the identity map.
    start = torch.tensor([[0.2, 0.2], [0.6, 0.2], [0.2, 0.6], [0.25, 0.25], [0.75, 0.75]])
    pixels = torch.full((5, 3, 2, 1, 1), 99.0)
    pixels[:, 0, :, 0, 0] = start
    action = torch.zeros(5, 3, 2)
    action[3:, 0] = torch.tensor([0.1, 0.0])
    action[3:, 1] = torch.tensor([0.0, 0.2])
    action[:, 2] = 1000.0  # The action after the last frame has no target.
    valid = torch.ones(5, 3, dtype=torch.bool)
    valid[4, 2] = False
    batch = {"pixels": pixels, "action": action, "valid": valid}
    extracted = extract(translation_model(), [batch], Passthrough(), rollout=True)
    future_changed = {**batch, "pixels": batch["pixels"].clone()}
    future_changed["pixels"][:, 1:] = -7.0
    torch.testing.assert_close(
        extract(translation_model(), [future_changed], Passthrough(), rollout=True)["predicted_z"],
        extracted["predicted_z"],
    )
    expected = start[:, None].repeat(1, 3, 1)
    expected[3:, 1] += torch.tensor([0.1, 0.0])
    expected[3:, 2] += torch.tensor([0.1, 0.2])
    torch.testing.assert_close(extracted["predicted_z"], expected)
    assert torch.all(extracted["z"][:, 1:] == 99.0)

    target = expected.clone()
    target[:3] *= torch.tensor([0.5, 1.0, 1.5])[None, :, None]
    target[3:, 1] += torch.tensor([0.03, 0.04])
    target[3:, 2] += torch.tensor([0.06, 0.08])
    target[4, 2] = 1000.0  # Padded labels must not affect scores.
    lat = Latents(
        **extracted,
        target=target.flip(-1),
        episode=torch.arange(5),
        train=torch.tensor([True, True, True, False, False]),
        target_names=["dot_y", "dot_x"],
        target_angle=[False, False],
        target_group=["state", "state"],
    )
    result = suite.run(lat, [{"name": "rollout_probe"}])["rollout_probe"]
    assert result["units"] == "coordinate side length"
    assert result["n_train"] == 9
    assert result["n_test"] == 5
    assert result["n_test_per_step"] == [2, 2, 1]
    np.testing.assert_allclose(result["rmse_per_step"], [0.0, 0.05, 0.1], atol=1e-6)
    np.testing.assert_allclose(result["mean_rmse"], 0.05, atol=1e-6)
    np.testing.assert_allclose(result["pooled_rmse"], np.sqrt(0.003), atol=1e-6)
    center_steps = np.sqrt([0.125, 0.1435, 0.009])
    np.testing.assert_allclose(result["center_control"]["rmse_per_step"], center_steps, atol=1e-6)
    np.testing.assert_allclose(
        result["center_control"]["mean_rmse"], center_steps.mean(), atol=1e-6
    )
    np.testing.assert_allclose(result["center_control"]["pooled_rmse"], np.sqrt(0.1092), atol=1e-6)
    np.testing.assert_allclose(result["r_pooled"], np.sqrt(0.003 / 0.1092), atol=1e-6)
    np.testing.assert_allclose(result["r"], 0.05 / center_steps.mean(), atol=1e-6)


def test_rollout_probe_reads_each_dot_from_its_own_targets():
    """A dot read through another dot's target columns, or an average over the wrong dots, would score
    the readable dot as unreadable. The predictions hold the action dot's position exactly and nothing
    of the fixed dot, whose targets are listed first and out of order. The training windows cross the
    four coordinates in a full factorial, so the readout's weights from the action features onto the
    fixed targets are exactly zero and the fixed dot is predicted by its training mean, 0.5."""
    rng = np.random.default_rng(0)
    steps = 2
    grid = np.array(np.meshgrid(*[[0.3, 0.7]] * 4)).reshape(4, -1).T  # 16 training windows
    test_action = rng.uniform(0.1, 0.9, (6, 2))
    test_fixed = rng.uniform(0.1, 0.9, (6, 2))
    action = np.concatenate([grid[:, :2], test_action])
    fixed = np.concatenate([grid[:, 2:], test_fixed])
    windows = len(action)
    action, fixed = (np.repeat(x[:, None], steps, 1) for x in (action, fixed))
    predicted = np.concatenate([action, np.zeros_like(fixed)], -1)
    target = np.stack([fixed[..., 0], action[..., 1], fixed[..., 1], action[..., 0]], -1)
    lat = Latents(
        z=torch.zeros(windows, steps, 1),
        action=torch.zeros(windows, steps, 2),
        valid=torch.ones(windows, steps, dtype=torch.bool),
        target=torch.tensor(target, dtype=torch.float32),
        episode=torch.arange(windows),
        train=torch.arange(windows) < 16,
        target_names=["fixed_x", "action_y", "fixed_y", "action_x"],
        target_angle=[False] * 4,
        target_group=["state"] * 4,
        predicted_z=torch.tensor(predicted, dtype=torch.float32),
    )
    dots = {"action": ["action_x", "action_y"], "stationary": ["fixed_x", "fixed_y"]}
    result = suite.run(lat, [{"name": "rollout_probe", "targets": dots}])["rollout_probe"]

    centre_fixed = np.sqrt(((test_fixed - 0.5) ** 2).sum(-1).mean())
    centre_action = np.sqrt(((test_action - 0.5) ** 2).sum(-1).mean())
    np.testing.assert_allclose(result["action"]["mean_rmse"], 0.0, atol=1e-5)
    np.testing.assert_allclose(
        result["action"]["center_control"]["mean_rmse"], centre_action, rtol=1e-5
    )
    np.testing.assert_allclose(result["stationary"]["mean_rmse"], centre_fixed, rtol=1e-4)
    np.testing.assert_allclose(result["stationary"]["r"], 1.0, rtol=1e-4)
    np.testing.assert_allclose(
        result["average"]["r"], centre_fixed / (centre_action + centre_fixed), rtol=1e-4
    )
