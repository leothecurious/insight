"""Draw frames from stored state rows with stable-envs.

A port of stable-envs' ``examples/lewm/state_render.py`` as a stable-pretraining GPU transform.
Each leaf of the frame spec names the dataset column and the ``[start, stop)`` columns that hold one
State field.
"""

import stable_envs as snv
import torch
from omegaconf import OmegaConf
from stable_envs import get_field
from torch import nn


class Frames(nn.Module):
    """Write ``(B, T, D)`` state columns into States and set ``batch["pixels"]`` to ``(B, T, 3, H, W)`` floats in [0, 1].

    Fields not named in ``spec["leaves"]`` keep their ``env.reset(seed=0)`` values, and the fields in
    ``spec["zero"]`` are cleared. ``overlays`` are applied in order to the uint8 frames, each as
    ``overlay(frames, batch) -> frames``.
    """

    def __init__(self, spec, overlays=()):
        super().__init__()
        if OmegaConf.is_config(spec):
            spec = OmegaConf.to_container(spec, resolve=True)
        self.spec = dict(spec)
        self.leaves = [(v["leaf"], v["column"], *v["columns"]) for v in spec["leaves"].values()]
        self.overlays = list(overlays)
        self._env = self._template = None
        self._states = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_env=None, _template=None, _states={})
        return state

    def _env_on(self, device):
        if self._env is None or self._env.device != device:
            if self._env is not None:
                self._env.close()
            self._env = snv.make(
                self.spec["env"], num_envs=1, device=str(device), **(self.spec.get("kwargs") or {})
            )
            self._env.reset(seed=0)
            self._template = self._env.get_state()
            self._states = {}
        return self._env

    def _state(self, n):
        if n not in self._states:
            self._states[n] = self._template.expand(n).clone()
        return self._states[n]

    @torch.no_grad()
    def forward(self, batch):
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
        for overlay in self.overlays:
            frames = overlay(frames, batch)
        h, w = frames.shape[1:3]
        batch["pixels"] = frames.permute(0, 3, 1, 2).reshape(b, t, 3, h, w).float().div_(255)
        return batch
