import numpy as np
from stable_worldmodel.data import HDF5Writer

EPISODES, STEPS = 10, 12


def agent_position(ep, step):
    """Pixel position of the agent: x moves 10 px per step, y 3 px per step and 12 px per episode."""
    return np.stack([30 + 10 * step, 40 + 12 * ep + 3 * step], axis=-1).astype(np.float32)


def write_toy_tworoom(path):
    """Write ``EPISODES`` TwoRoom episodes of ``STEPS`` rows; the first action of an episode is NaN."""
    steps = np.arange(STEPS)
    action = np.zeros((STEPS, 2), np.float32)
    action[0] = np.nan
    episodes = [
        {
            "action": action,
            "pos_agent": agent_position(ep, steps),
            "ep_idx": np.full(STEPS, ep, np.int32),
            "step_idx": steps.astype(np.int64),
        }
        for ep in range(EPISODES)
    ]
    with HDF5Writer(path) as writer:
        writer.write_episodes(episodes)
