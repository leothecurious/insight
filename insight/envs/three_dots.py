"""The 3-dots environment of Sobal et al. (2022, arXiv 2211.10831, App. A.6), as a stable-envs environment.

Three dots, each drawn on its own colour channel with the single dot's geometry (``moving_dot.py``):
red is moved by the episode's actions, green by actions of its own that the model never receives, and
blue stays where the episode started it. The authors' code (vladisai/JEPA_SSL_NeurIPS_2022,
``data/multiple.py``) builds the green dot as a second ``ContinuousMotionDataset`` and the blue dot as a
``DeterministicMotionDataset`` with ``max_step=0``, and feeds the model only the first dot's actions
(``probing.py``: "drop actions of other spheres").

The green dot's heading is drawn once per episode and kept as a hidden State field; each step then draws
its direction from a Von Mises around that heading and its norm uniformly, like ``VonMisesHeading`` does
for a whole sequence. Both come from the environment's seeded generator, so the same seed replays the
same dot.
"""

import numpy as np
import stable_envs as snv
import torch
from stable_envs import spaces
from stable_envs.registry import register

from insight.envs.moving_dot import MARGIN, MAX_STEP, SIGMA

CONCENTRATION = 0.2  # the code's Von Mises concentration of the moving dot (paper: 1.3)
COLORS = {"action_dot": (255, 0, 0), "random_dot": (0, 255, 0), "fixed_dot": (0, 0, 255)}


@register
class ThreeDots(snv.Env):
    """The state is the positions of the action, random and fixed dots, ``(x, y)`` each."""

    family = "toy"
    state_dim = 6
    state_names = (
        "action_dot_x",
        "action_dot_y",
        "random_dot_x",
        "random_dot_y",
        "fixed_dot_x",
        "fixed_dot_y",
    )
    state_layout = (
        (("action_dot", "position"), 2, 0),
        (("random_dot", "position"), 2, 2),
        (("fixed_dot", "position"), 2, 4),
        (("random_dot", "heading"), 1, None),
    )
    action_dim = 2  # the action dot's move; the random dot's moves are drawn inside the environment
    action_low, action_high = -MAX_STEP, MAX_STEP
    default_size = 32
    MARGIN = MARGIN
    action_leaf = "action_dot.position"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._heading = torch.zeros(self.B, device=self.device)

    def make_variation_space(self):
        dots = {
            name: spaces.Dict({"color": spaces.RGBBox(init_value=color)})
            for name, color in COLORS.items()
        }
        sigma = spaces.Box(low=0.01, high=0.2, shape=(1,), init_value=[SIGMA])
        return spaces.Dict({**dots, "sigma": sigma})

    def sample_states(self, batch, rng=None):
        rng = self.rng if rng is None else rng
        return rng.uniform(MARGIN, 1 - MARGIN, size=(batch, 6)).astype(np.float32)

    def reset(self, seed=None):
        self._reseed(seed)
        heading = self.rng.uniform(0, 2 * np.pi, size=self.B)
        self._heading = torch.as_tensor(heading, dtype=torch.float32, device=self.device)
        return super().reset()

    def _hidden_state(self):
        return {("random_dot", "heading"): self._heading[: self._live_rows]}

    def _restore_state(self, state):
        self._heading[: len(state)] = state["random_dot", "heading"].reshape(-1)
        self._set_render_state(self._pack_render_state(state))

    def _step(self, actions, *, active=None):
        rows = self._get_render_state()
        n = len(rows)
        theta = self.rng.vonmises(self._heading[:n].cpu().numpy().astype(np.float64), CONCENTRATION)
        norm = self.rng.uniform(0, MAX_STEP, size=n)
        random_move = torch.as_tensor(
            (norm[:, None] * np.stack([np.cos(theta), np.sin(theta)], -1)).astype(np.float32),
            device=self.device,
        )
        move = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if active is not None:
            move, random_move = move * active[:, None], random_move * active[:, None]
        rows[:, :2] = (rows[:, :2] + move).clamp(MARGIN, 1 - MARGIN)
        rows[:, 2:4] = (rows[:, 2:4] + random_move).clamp(MARGIN, 1 - MARGIN)
        return snv.StepResult({"state": self.get_state()}, batch_size=[self.num_envs])

    @torch.no_grad()
    def _render(
        self, states=None, size=None, supersample=None, variation=None, style=None, out=None
    ):
        """Frames ``(..., size, size, 3)`` uint8 of ``(..., 6)`` rows; each dot's bump fills its own channel."""
        lead = states.shape[:-1]
        pos = states.reshape(-1, 3, 2)
        sty = self.resolve_style(style, variation, batch=len(pos))
        size = size or self.size
        grid = (torch.arange(size, dtype=pos.dtype, device=pos.device) + 0.5) / size
        two_var = 2 * sty["sigma"].square()[:, None, None]
        bump_x = torch.exp(-(grid - pos[:, :, :1]).square() / two_var)
        bump_y = torch.exp(-(grid - pos[:, :, 1:]).square() / two_var)
        bump = bump_y[:, :, :, None] * bump_x[:, :, None, :]
        color = torch.stack([sty[f"{name}.color"] for name in COLORS], 1)
        frames = (bump[..., None] * color[:, :, None, None, :]).sum(1).round().clamp(0, 255)
        frames = frames.to(torch.uint8).reshape(*lead, *frames.shape[1:])
        if out is not None:
            return out.copy_(frames)
        return frames
