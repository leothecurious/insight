"""The moving dot of Sobal et al. (2022, arXiv 2211.10831), as a stable-envs environment.

A dot in the unit square moves by its action and stops ``MARGIN`` short of every edge. Frames are a
Gaussian bump of peak ``dot.color`` (white) on black, drawn analytically at any size. Lengths are in
units of the square's side. The geometry follows the authors' code (vladisai/JEPA_SSL_NeurIPS_2022,
``data/single.py``) where it differs from the paper's text: its pixel values on a 28 px frame divided
by 27, the span of its pixel coordinates.
"""

import numpy as np
import stable_envs as snv
import torch
from stable_envs import spaces
from stable_envs.registry import register

MAX_STEP = 4 / 27  # largest action norm; 4 px in the code (paper: 0.14)
SIGMA = 1.3 / 27  # Gaussian blur of the dot; 1.3 px in the code (paper: 0.05)
# The dot starts and stays 2 sigma from every edge, as in the code (paper: clipped to [0, 1]).
MARGIN = 2 * SIGMA


@register
class MovingDot(snv.Env):
    """A dot moved by its action; the state is its position ``(x, y)``."""

    family = "toy"
    state_dim = 2
    state_names = ("dot_x", "dot_y")
    state_layout = ((("dot", "position"), 2, 0),)
    action_dim = 2
    action_leaf = "dot.position"  # the State field the action moves
    action_low, action_high = -MAX_STEP, MAX_STEP
    default_size = 32
    MARGIN = MARGIN

    def make_variation_space(self):
        return spaces.Dict(
            {
                "dot": spaces.Dict(
                    {
                        "color": spaces.RGBBox(init_value=(255, 255, 255)),
                        "sigma": spaces.Box(low=0.01, high=0.2, shape=(1,), init_value=[SIGMA]),
                    }
                )
            }
        )

    def sample_states(self, batch, rng=None):
        rng = self.rng if rng is None else rng
        return rng.uniform(MARGIN, 1 - MARGIN, size=(batch, 2)).astype(np.float32)

    def _step(self, actions, *, active=None):
        rows = self._get_render_state()
        move = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if active is not None:
            move = move * active[:, None]
        rows.copy_((rows + move).clamp(MARGIN, 1 - MARGIN))
        return snv.StepResult({"state": self.get_state()}, batch_size=[self.num_envs])

    @torch.no_grad()
    def _render(
        self, states=None, size=None, supersample=None, variation=None, style=None, out=None
    ):
        """Frames ``(..., size, size, 3)`` uint8 of ``(..., 2)`` rows; the bump is analytic, so any ``size`` works."""
        lead = states.shape[:-1]
        pos = states.reshape(-1, 2)
        sty = self.resolve_style(style, variation, batch=len(pos))
        size = size or self.size
        grid = (torch.arange(size, dtype=pos.dtype, device=pos.device) + 0.5) / size
        two_var = 2 * sty["dot.sigma"].square()[:, None]
        bump_x = torch.exp(-(grid - pos[:, :1]).square() / two_var)
        bump_y = torch.exp(-(grid - pos[:, 1:]).square() / two_var)
        bump = bump_y[:, :, None] * bump_x[:, None, :]
        frames = (bump[..., None] * sty["dot.color"][:, None, None, :]).round().to(torch.uint8)
        frames = frames.reshape(*lead, *frames.shape[1:])
        if out is not None:
            return out.copy_(frames)
        return frames
