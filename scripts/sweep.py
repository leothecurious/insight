"""One hyperparameter trial: train a run, score it on a validation split, return the score.

``uv run scripts/sweep.py -m +experiment=sobal_dot/reconstruction_clean +sweep=reconstruction`` hands this
function to Hydra's Optuna sweeper (``configs/sweep/``), which draws each trial's overrides from the
search space and keeps nothing but the returned value. The score is the ``r`` of ``rollout_probe``
(mean per-step RMSE over that of predicting the centre) with the probe fit on the probe file and scored on
the validation file of ``sweep.score_files``, so selection never reads the test file.
"""

import json
import math
import shutil
import time
from pathlib import Path

import hydra
import stable_worldmodel as swm
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from measure import measure
from omegaconf import OmegaConf
from train import train

from insight.checkpoints import filename as checkpoint_filename

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def score(cfg):
    """Metrics of the finished run on the validation file of ``sweep.score_files``."""
    overrides = [
        f"run={cfg.run}",
        f"weights={checkpoint_filename(cfg.series, cfg.trainer.max_epochs, cfg.trainer.max_epochs)}",
        f"files={OmegaConf.to_container(cfg.sweep.score_files)}",
        "wandb=false",
        # Selection reads r only.
        "metrics=[{name:rollout_probe}]",
        *(f"+experiment={name}" for name in cfg.sweep.measure_experiments),
    ]
    if cfg.paths.data:
        overrides.append(f"paths.data={cfg.paths.data}")
    if cfg.paths.runs:
        overrides.append(f"paths.runs={cfg.paths.runs}")
    # Hydra allows one composition at a time; the trial's own is complete.
    GlobalHydra.instance().clear()
    with initialize_config_dir(str(CONFIGS), version_base=None):
        return measure(compose("measure", overrides=overrides))


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def run(cfg):
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir = Path(swm.data.utils.get_cache_dir(cfg.paths.runs, "checkpoints"), cfg.run).resolve()
    start = time.perf_counter()
    try:
        train(cfg)
        trained = time.perf_counter()
        probe = score(cfg)["rollout_probe"]
    finally:
        # The weights, latents and checkpoints of a trial are not kept; its record is trial.json.
        if not cfg.sweep.keep_runs:
            shutil.rmtree(run_dir, ignore_errors=True)
    (output_dir / "trial.json").write_text(
        json.dumps(
            {
                "run": cfg.run,
                "r": probe["r"],
                "rmse_per_step": probe["rmse_per_step"],
                "train_seconds": round(trained - start, 1),
                "measure_seconds": round(time.perf_counter() - trained, 1),
                "overrides": list(HydraConfig.get().overrides.task),
            },
            indent=1,
        )
    )
    # Optuna fails a trial on NaN; a diverged run ranks last instead.
    return probe["r"] if math.isfinite(probe["r"]) else float("inf")


if __name__ == "__main__":
    run()
