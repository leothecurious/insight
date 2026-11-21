"""Measure a trained model: ``uv run scripts/measure.py run=<run> weights=weights_epoch_N.pt``.

Ported from le-wm 189a36c measure.py. The run's ``config.yaml`` gives the dataset, the frames and the
episode split; windows of its training and validation episodes are drawn through the run's frames,
encoded once, and every metric reads those latents. The pair metric and the colour probe stamp their
own tag on dataset frames (``insight.metrics.pairs``). Results are appended to
``<run>/measure_results.jsonl`` and, when the run used W&B, written to its summary.
"""

import json
import time
from pathlib import Path

import hydra
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf

from insight.data.build import build
from insight.metrics import data, pairs, predictor, stimuli, suite
from insight.metrics.latents import Latents, extract


def get_latents(cfg, train, model, path, train_set, val_set):
    if cfg.latents_path:
        return Latents.load(cfg.latents_path)

    frame_columns = {leaf.column for leaf in train.data.frames.leaves.values()}
    if train.data.frames.get("nuisance"):
        frame_columns |= set(train.data.ids.values())
    frame_columns = sorted(frame_columns)
    loader, target, episode, is_train = data.windows_loader(
        train_set, val_set, frame_columns, train.data.measure, cfg.windows, cfg.seed
    )
    targets = train.data.measure.targets
    out = extract(model, loader, train_set.dataset.gpu_transform, cfg.windows.precision)
    lat = Latents(
        **out,
        target=torch.from_numpy(target),
        episode=torch.from_numpy(episode),
        train=torch.from_numpy(is_train),
        target_names=[t.name for t in targets],
        target_angle=[bool(t.get("angle", False)) for t in targets],
        target_group=[t.get("group", "state") for t in targets],
    )
    lat.save(path)
    return lat


def stamping(cfg, train, leaves=None):
    return pairs.stamping(
        OmegaConf.to_container(train.data, resolve=True), train.seed, cfg.tag_size, leaves
    )


def measure_pairs(cfg, train, model, ds):
    """Pair cosines: dataset frames stamped with the tag's colours, or the t_position scenes."""
    if cfg.pairs.suite == "t_position":
        frames, tag = stamping(cfg, train, stimuli.GOAL_LEAVES)
        return pairs.measure_t_position(model, frames, cfg.pairs.n, tag, cfg.device, cfg.seed)
    frames, tag = stamping(cfg, train)
    sample = pairs.sample_frames(frames, ds, cfg.pairs.n, cfg.seed, cfg.device)
    return pairs.measure(model, sample, tag, cfg.seed)


def measure_colour(cfg, train, model, ds):
    """Colour probe on dataset frames, each stamped with its own colour."""
    frames, tag = stamping(cfg, train)
    sample = pairs.sample_frames(frames, ds, cfg.colour.n, cfg.seed, cfg.device)
    return pairs.measure_colour(
        model, sample, tag, cfg.colour.kinds, cfg.colour.train_fraction, cfg.seed
    )


@hydra.main(version_base=None, config_path="../configs", config_name="measure")
def run(cfg):
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
    train_set, val_set = build(train.data, train.seed)
    lat = get_latents(cfg, train, model, latents_path, train_set, val_set)
    metrics = suite.run(lat, OmegaConf.to_container(cfg.metrics, resolve=True))
    if cfg.get("pred_copy"):
        metrics["pred_copy"] = predictor.measure(model, lat, **cfg.pred_copy)
    if cfg.get("pairs"):
        metrics["pairs"] = measure_pairs(cfg, train, model, train_set.dataset)
    if cfg.get("colour"):
        metrics["colour_probe"] = measure_colour(cfg, train, model, train_set.dataset)
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
