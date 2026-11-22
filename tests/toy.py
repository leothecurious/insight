from pathlib import Path

from hydra import compose, initialize_config_dir

from insight.data.build import build
from insight.data.collect import collect
from insight.data.frames import Frames
from insight.policies.von_mises import VonMisesHeading

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SEQUENCES, STEPS = 8, 5


def build_toy_dot(tmp_path, monkeypatch, overrides=()):
    """Collect a tiny train and validation set of the moving dot and return ``build``'s training set.

    The files go through ``collect`` (h5py) and ``swm.load_dataset`` like a real dataset.
    """
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path))
    with initialize_config_dir(str(CONFIGS), version_base=None):
        cfg = compose("train", overrides=["+experiment=sobal_dot/lewm_clean", *overrides])
    policy = VonMisesHeading()
    frames = Frames(cfg.data.frames, cfg.data.ids)
    for i, name in enumerate((cfg.data.dataset.name, cfg.data.split.val.name)):
        path = tmp_path / "datasets" / name
        collect(path, frames, cfg.data.ids, policy, SEQUENCES, STEPS, i, i * SEQUENCES, SEQUENCES)
    return build(cfg.data, seed=0)[0]
