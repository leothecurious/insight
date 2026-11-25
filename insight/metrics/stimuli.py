"""Push-T scenes with a controllable block pose and goal outline, for the t_position pair suite.

Ported from Enigma-Incorporated/The-Obsessed-Encoder@37819425 (``leworldmodel/additional_files/stimuli.py``,
MIT licence). The scenes are drawn by stable-envs PushT through
``Frames``, with the goal pose written into the same State leaves as in ``configs/data/pusht_randgoal.yaml``.
"""

import numpy as np
import torch

GOAL_LEAVES = {
    "goal_xy": {
        "leaf": "owned.variation.goal.position.current.value",
        "column": "goal_pose",
        "columns": [0, 2],
    },
    "goal_angle": {
        "leaf": "owned.variation.goal.angle.current.value",
        "column": "goal_pose",
        "columns": [2, 3],
    },
}

# p1..p99 of the pusht_expert_train state column [agent_xy].
AGENT_LO = np.array([38.0, 75.0])
AGENT_HI = np.array([449.0, 482.0])

# A tee whose centre lies in [140, 372]^2 stays inside the walls at any angle.
# Used for the block and for the goal outline.
TEE_LO = np.array([140.0, 140.0])
TEE_HI = np.array([372.0, 372.0])


def sample_orig(rng, k):
    """``k`` state rows [agent_xy, block_xy, block_angle, velocity_xy = 0], as float32 ``(k, 1, 7)``."""
    agent = rng.uniform(AGENT_LO, AGENT_HI, size=(k, 2))
    block = rng.uniform(TEE_LO, TEE_HI, size=(k, 2))
    angle = rng.uniform(0.0, 2 * np.pi, size=(k, 1))
    state = np.concatenate([agent, block, angle, np.zeros((k, 2))], axis=1)
    return torch.from_numpy(state[:, None]).float()


def sample_dest(rng, k):
    """``k`` goal poses [x, y, angle], as float32 ``(k, 1, 3)``."""
    xy = rng.uniform(TEE_LO, TEE_HI, size=(k, 2))
    angle = rng.uniform(0.0, 2 * np.pi, size=(k, 1))
    return torch.from_numpy(np.concatenate([xy, angle], axis=1)[:, None]).float()
