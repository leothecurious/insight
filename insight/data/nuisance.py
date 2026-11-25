"""Nuisance overlays: seeded axes of a dataset, drawn on the uint8 frames right after ``env.render``.

``CornerSquare`` ports the corner tag of Enigma-Incorporated/The-Obsessed-Encoder@37819425
(``pixel_tag.py``). ``UniformNoise`` implements the uniform pixel noise described by Sobal et al. (2022,
arXiv 2211.10831, App. A.2). Their parameters are pure functions of ``(seed, episode)`` or
``(seed, episode, step)``, so train, measure and eval draw the same overlay for the same dataset row,
independent of how rows are batched. The hash is the counter-based ``lowbias32`` of stable-envs
``envs/miniworld/simulation/rng.py`` (torch version), which is private to its family.
"""

import torch

_M32 = 0xFFFFFFFF
_C1, _C2 = 0x7FEB352D, 0x846CA68B


def _mul(x, c):
    """``x * c mod 2**32`` for ``x, c < 2**32`` in int64 without overflow."""
    return (x * (c & 0xFFFF) + (((x * (c >> 16)) & 0xFFFF) << 16)) & _M32


def mix(x):
    """The ``lowbias32`` integer hash of an int or an int64 tensor."""
    x = x & _M32
    x = x ^ (x >> 16)
    x = _mul(x, _C1)
    x = x ^ (x >> 15)
    x = _mul(x, _C2)
    return x ^ (x >> 16)


def draw(seed, episode, step, k):
    """32 random bits for draw ``k`` of the stream of ``(seed, episode[, step])``, as int64.

    ``seed`` is an int; ``episode`` and ``step`` are int64 tensors of one shape, or ``step`` is None; ``k``
    is an int or an int64 tensor that broadcasts against them.
    """
    base = mix(mix(mix((seed & _M32) ^ 0x3C6EF372) ^ ((seed >> 32) & _M32)) ^ (episode & _M32))
    if step is not None:
        base = mix(base ^ mix(step))
    return mix(base ^ mix(k))


class CornerSquare:
    """A square of ``size`` pixels in the top-left corner, in one colour per episode or per frame.

    ``vary`` is ``episode`` (the colour is constant within an episode, so it is predictable from the
    past) or ``frame`` (a new colour on every dataset row, which the past cannot predict).
    """

    def __init__(self, size, vary, seed):
        assert vary in ("episode", "frame"), vary
        self.size, self.vary, self.seed = size, vary, seed

    def colour(self, episode, step=None):
        """RGB uint8 colours of ``(..., 3)`` for the int tensors of dataset ids ``episode`` and ``step``."""
        step = step.long() if self.vary == "frame" else None
        rgb = [draw(self.seed, episode.long(), step, channel) >> 24 for channel in range(3)]
        return torch.stack(rgb, -1).to(torch.uint8)

    def __call__(self, frames, episode, step=None):
        """Stamp ``(..., H, W, 3)`` uint8 ``frames`` in place and return them."""
        frames[..., : self.size, : self.size, :] = self.colour(episode, step)[..., None, None, :]
        return frames


class UniformNoise:
    """Blend uniform noise into every frame: ``(o + alpha * 255 * Z) / (1 + alpha)`` with Z in [0, 1).

    Z is drawn per pixel and shared by the three channels. ``vary`` is ``episode`` (one pattern per
    episode, so it is predictable from the past) or ``frame`` (a new pattern on every dataset row).
    Dividing by ``1 + alpha`` keeps the frame in uint8 range at the paper's signal-to-noise ratio.
    """

    def __init__(self, alpha, vary="frame", seed=0):
        assert vary in ("episode", "frame"), vary
        self.alpha, self.vary, self.seed = alpha, vary, seed

    def pattern(self, episode, step, height, width):
        """Z of ``(..., H, W)`` for the int tensors of dataset ids ``episode`` and ``step``."""
        pixel = torch.arange(height * width, device=episode.device)
        step = step.long()[..., None] if self.vary == "frame" else None
        bits = draw(self.seed, episode.long()[..., None], step, pixel)
        return (bits >> 8).float().div_(1 << 24).reshape(*episode.shape, height, width)

    def __call__(self, frames, episode, step=None):
        """Blend ``(B, T, H, W, 3)`` uint8 ``frames`` and return them."""
        z = self.pattern(episode, step, *frames.shape[-3:-1])
        return (
            ((frames.float() + self.alpha * 255 * z[..., None]) / (1 + self.alpha))
            .round()
            .to(torch.uint8)
        )
