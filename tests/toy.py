from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from stable_worldmodel.data import HDF5Writer

from insight.data.build import build

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
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


def build_toy_tworoom(tmp_path, monkeypatch, overrides=()):
    """Write a toy TwoRoom file and return ``build``'s training set of it, at frameskip 2."""
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    path = tmp_path / "toy.h5"
    write_toy_tworoom(path)
    with initialize_config_dir(str(CONFIGS), version_base=None):
        cfg = compose("train", overrides=["data=tworoom", *overrides])
    cfg.data.dataset.name = str(path)
    cfg.data.dataset.frameskip = 2
    return build(cfg.data, seed=0)[0]
