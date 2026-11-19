"""Measure a trained model: ``uv run scripts/measure.py run=<run> weights=weights_epoch_N.pt``.

Ported from le-wm 189a36c measure.py. The run's ``config.yaml`` gives the dataset, the frames and the
episode split; windows of its training and validation episodes are drawn through the run's frames,
encoded once, and every metric reads those latents. Results are appended to ``<run>/measure_results.jsonl``
and, when the run used W&B, written to its summary.
"""

import json
import time
from pathlib import Path

import hydra
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from insight.data.build import build
from insight.metrics import data, predictor, suite
from insight.metrics.latents import Latents, extract


def get_latents(cfg, train, model, path):
    if cfg.latents_path:
        return Latents.load(cfg.latents_path)

    train_set, val_set = build(train.data, train.seed)
    frame_columns = sorted({leaf.column for leaf in train.data.frames.leaves.values()})
    loader, target, episode, is_train = data.build(
        train_set, val_set, frame_columns, train.data.measure, cfg.windows, cfg.seed
    )
    names, angle, group = data.target_meta(train.data.measure.targets)
    out = extract(model, loader, train_set.dataset.gpu_transform, cfg.windows.precision)
    lat = Latents(
        **out,
        target=torch.from_numpy(target),
        episode=torch.from_numpy(episode),
        train=torch.from_numpy(is_train),
        target_names=names,
        target_angle=angle,
        target_group=group,
    )
    lat.save(path)
    return lat


@hydra.main(version_base=None, config_path="../configs", config_name="measure")
def run(cfg):
    # The sklearn probes otherwise take every core of a shared host.
    threadpool_limits(cfg.threads)
    torch.set_num_threads(cfg.threads)
    run_dir = Path(swm.data.utils.get_cache_dir(cfg.paths.runs, sub_folder="checkpoints"), cfg.run)
    run_dir = run_dir.resolve()
    train = OmegaConf.load(run_dir / "config.yaml")
    train.data.dataset.cache_dir = cfg.paths.data

    if cfg.weights == "random":
        torch.manual_seed(cfg.seed)
        model = hydra.utils.instantiate(train.model)
    else:
        model = swm.wm.utils.load_pretrained(f"{cfg.run}/{cfg.weights}", cache_dir=cfg.paths.runs)
    model = model.to(cfg.device).eval().requires_grad_(False)
    stem = Path(cfg.weights).stem
    latents_path = run_dir / f"{stem}_latents.pt"

    start = time.perf_counter()
    lat = get_latents(cfg, train, model, latents_path)
    metrics = suite.run(lat, OmegaConf.to_container(cfg.metrics, resolve=True))
    if cfg.get("pred_copy"):
        metrics["pred_copy"] = predictor.measure(model, lat, **cfg.pred_copy)
    seconds = time.perf_counter() - start

    print(json.dumps(metrics, indent=1))
    record = {
        "weights": cfg.weights,
        "latents": str(cfg.latents_path or latents_path),
        "measure_seconds": round(seconds, 1),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics": metrics,
    }
    with (run_dir / "measure_results.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")

    if cfg.wandb and train.wandb.enabled:
        import wandb

        with wandb.init(**OmegaConf.to_container(train.wandb.config, resolve=True)) as wb:
            wb.summary[f"measure/{stem}"] = metrics


if __name__ == "__main__":
    run()
