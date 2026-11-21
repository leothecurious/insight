"""Draw frames from stored state rows with stable-envs.

A port of stable-envs' ``examples/lewm/state_render.py`` as a stable-pretraining GPU transform.
Each leaf of the frame spec names the dataset column and the ``[start, stop)`` columns that hold one
State field. The overlays of ``spec["nuisance"]`` (``insight.data.nuisance``) are drawn on the rendered frames.
"""

import hydra
import stable_envs as snv
import torch
from stable_envs import get_field
from torch import nn

from insight import envs  # noqa: F401  registers the toy environments


class Frames(nn.Module):
    """Write ``(B, T, D)`` state columns into States and set ``batch["pixels"]`` to ``(B, T, 3, H, W)`` floats in [0, 1].

    Fields not named in ``spec["leaves"]`` keep their ``env.reset(seed=0)`` values, and the fields in
    ``spec["zero"]`` are cleared. ``spec`` is a plain container. The overlays of ``spec["nuisance"]``,
    applied in order, read the dataset's ``ids`` columns (``{"episode": name, "step": name}``) of the batch.
    """

    def __init__(self, spec, ids):
        super().__init__()
        self.spec = spec
        self.ids = ids
        self.nuisance = [hydra.utils.instantiate(c) for c in (spec.get("nuisance") or {}).values()]
        self.leaves = [(v["leaf"], v["column"], *v["columns"]) for v in spec["leaves"].values()]
        self._env = self._template = None
        self._states = {}

    def __getstate__(self):
        # gymnasium deep-copies env kwargs (Wrapper.spec); drop the live env from copies.
        state = self.__dict__.copy()
        state.update(_env=None, _template=None, _states={})
        return state

    def _env_on(self, device):
        if self._env is None:
            self._env = snv.make(
                self.spec["env"], num_envs=1, device=str(device), **(self.spec.get("kwargs") or {})
            )
            self._env.reset(seed=0)
            self._template = self._env.get_state()
        return self._env

    def _state(self, n):
        if n not in self._states:
            self._states[n] = self._template.expand(n).clone()
        return self._states[n]

    @torch.no_grad()
    def draw(self, batch):
        """Return ``(B, T, H, W, 3)`` uint8 frames of the ``(B, T, D)`` columns of ``batch``.

        With nuisance overlays, ``batch`` also holds the ``(B, T)`` id columns.
        """
        column = self.leaves[0][1]
        b, t = batch[column].shape[:2]
        env = self._env_on(batch[column].device)
        state = self._state(b * t)
        for leaf, column, start, stop in self.leaves:
            field = get_field(state, leaf)
            field.copy_(batch[column][..., start:stop].reshape(field.shape).to(field.dtype))
        for leaf in self.spec.get("zero") or ():
            get_field(state, leaf).zero_()
        frames = env.render(state, size=self.spec["size"])
        frames = frames.reshape(b, t, *frames.shape[-3:])
        for overlay in self.nuisance:
            frames = overlay(frames, batch[self.ids["episode"]], batch[self.ids["step"]])
        return frames

    def columns(self, state):
        """Read the columns of a State, shaped ``(B, 1, D)``: the inverse of what ``draw`` writes."""
        out = {}
        for leaf, column, start, stop in self.leaves:
            field = get_field(state, leaf)
            if column not in out:
                width = max(stop for _, c, _, stop in self.leaves if c == column)
                out[column] = field.new_zeros(field.shape[0], 1, width)
            out[column][..., start:stop] = field.reshape(field.shape[0], 1, stop - start)
        return out

    def forward(self, batch):
        batch["pixels"] = self.draw(batch).permute(0, 1, 4, 2, 3).float().div_(255)
        return batch
