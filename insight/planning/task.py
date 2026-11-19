"""Adapter that lets stable-worldmodel's ``World`` step a stable-envs environment.

A port of the ``Task`` of stable-envs ``examples/lewm/eval.py``.
Observations and the goal image are drawn by the run's ``Frames``, so the planner sees the frames of
training.
"""

from typing import ClassVar

import gymnasium as gym
import numpy as np
import stable_envs as snv
import torch

ENV_ID = "insight/Task-v0"


class Task(gym.Env):
    """One stable-envs environment with the stable-worldmodel task interface.

    Args:
        frames: The run's ``insight.data.frames.Frames``. Its env and kwargs also build the simulator.
        spec: The ``eval`` block of the run's data config.
        device: Device of the simulator.
        frames_device: Device that ``frames`` draws on.
    """

    metadata: ClassVar[dict] = {"render_modes": ["rgb_array"]}

    def __init__(self, frames, spec, device="cpu", frames_device="cuda", render_mode="rgb_array"):
        self.render_mode = render_mode
        self.frames = frames
        self.frames_device = frames_device
        self.env = snv.make(
            frames.spec["env"], num_envs=1, device=device, **(frames.spec.get("kwargs") or {})
        )
        self.observation_leaf = spec["observation_leaf"]
        self.proprio = list(spec["proprio_columns"])
        self.reset_columns = dict(spec["reset_options"])
        action = self.env.action_spec
        low, high = (
            np.broadcast_to(np.asarray(v, np.float32), action.shape)
            for v in (action.low, action.high)
        )
        self.action_space = gym.spaces.Box(low, high, dtype=np.float32)
        dim = int(snv.get_field(self.env.reset(seed=0), self.observation_leaf).shape[-1])
        box = lambda n: gym.spaces.Box(-np.inf, np.inf, (n,), np.float64)
        self.observation_space = gym.spaces.Dict(
            {"proprio": box(len(self.proprio)), "state": box(dim)}
        )
        self._goal_columns = None

    def reset_options_from_dataset(self, init_row, goal_row):
        """Map dataset columns to reset options; ``World`` calls this before ``reset``."""
        rows = {**init_row, **goal_row}
        self._goal_columns = {
            leaf["column"]: goal_row[f"goal_{leaf['column']}"]
            for leaf in self.frames.spec["leaves"].values()
        }
        return {
            name: np.asarray(rows[column], np.float64)
            for name, column in self.reset_columns.items()
        }

    def _draw(self, columns):
        batch = {k: v.to(self.frames_device) for k, v in columns.items()}
        return self.frames.draw(batch)[0].cpu().numpy()

    def _observation(self, state):
        obs = (
            snv.get_field(state, self.observation_leaf)
            .reshape(-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        return {"proprio": obs[self.proprio], "state": obs}

    def _info(self):
        return {"goal": self._goal}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._state = self.env.reset(seed=seed, options=options)
        goal = {
            k: torch.as_tensor(v, dtype=torch.float32)[None, None]
            for k, v in self._goal_columns.items()
        }
        self._goal = self._draw(goal)
        return self._observation(self._state), self._info()

    def step(self, action):
        act = torch.as_tensor(
            np.asarray(action), dtype=self.env.action_spec.dtype, device=self.env.device
        )
        out = self.env.step(act.reshape(1, -1), frames=None)
        self._state = out["state"]
        reward = float(out["reward"][0])
        terminated, truncated = bool(out["terminated"][0]), bool(out["truncated"][0])
        return self._observation(self._state), reward, terminated, truncated, self._info()

    def render(self):
        return self._draw(self.frames.columns(self._state))

    def close(self):
        self.env.close()


gym.register(ENV_ID, entry_point=Task)
