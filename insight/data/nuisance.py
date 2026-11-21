"""Nuisance overlays: seeded axes of a dataset, drawn on the uint8 frames right after ``env.render``.

``CornerSquare`` ports the corner tag of Enigma-Incorporated/The-Obsessed-Encoder@37819425
(``pixel_tag.py``), which the le-wm fork ``tag.py`` carried over. Its colour is a pure function of
``(seed, episode)`` or ``(seed, episode, step)``, so train, measure and eval draw the same colour for the
same dataset row, independent of how rows are batched. The hash is the counter-based ``lowbias32`` of
stable-envs ``envs/miniworld/simulation/rng.py`` (torch version), which is private to its family.
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

    ``seed`` and ``k`` are ints; ``episode`` and ``step`` are int64 tensors of one shape, or ``step`` is None.
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

    def __init__(self, size=5, vary="episode", seed=0):
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
