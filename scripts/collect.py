"""Record a dataset of state rows: ``uv run scripts/collect.py data=dot name=dot_train sequences=90000 seed=0``.

The frame spec of the ``data`` config names the environment and maps its State to dataset columns; the
``policy`` group supplies the actions. The file is written to ``<paths.data>/datasets/<name>.h5``.
"""

from pathlib import Path

import hydra
import stable_worldmodel as swm

from insight.data.collect import collect
from insight.data.frames import Frames


@hydra.main(version_base=None, config_path="../configs", config_name="collect")
def run(cfg):
    path = Path(swm.data.utils.get_cache_dir(cfg.paths.data, "datasets"), f"{cfg.name}.h5")
    collect(
        path,
        Frames(cfg.data.frames, cfg.data.ids),
        cfg.data.ids,
        hydra.utils.instantiate(cfg.policy),
        cfg.sequences,
        cfg.steps,
        cfg.seed,
        cfg.first_episode,
        cfg.chunk,
        cfg.device,
    )
    print(f"wrote {cfg.sequences} sequences to {path}")


if __name__ == "__main__":
    run()
