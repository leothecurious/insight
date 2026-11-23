"""Plan with a trained model: ``uv run scripts/eval.py run=<run> weights=weights_epoch_N.pt``.

Follows stable-worldmodel ``scripts/plan/eval_wm.py`` and stable-envs ``examples/lewm/eval.py``. The
run's ``config.yaml`` gives the dataset, the frames, the action statistics and the planning units.
``weights=random`` evaluates random actions.
"""

import dataclasses
import json
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf
from torchvision.transforms import v2 as transforms

from insight.data.build import scalers
from insight.data.frames import Frames
from insight.planning.task import ENV_ID


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def run(cfg):
    run_dir = Path(swm.data.utils.get_cache_dir(cfg.paths.runs, sub_folder="checkpoints"), cfg.run)
    run_dir = run_dir.resolve()
    train = OmegaConf.load(run_dir / "config.yaml")
    train.data.dataset.cache_dir = cfg.paths.data
    history_len = train.wm.history_size if cfg.history_len is None else cfg.history_len
    plan = swm.PlanConfig(
        **cfg.plan_config,
        action_block=train.data.dataset.frameskip,
        history_len=history_len,
    )
    assert plan.plan_len <= cfg.eval_budget, "the plan must fit in eval_budget"

    # Start steps that have a goal goal_offset_steps later in the same episode, as eval_wm.py draws them.
    # The dataset is read raw: at the dataset's own frameskip and without the training transform.
    ds = train.data.dataset
    dataset = swm.data.load_dataset(
        ds.name,
        cache_dir=cfg.paths.data,
        keys_to_load=list(ds.keys_to_load),
        keys_to_cache=list(ds.keys_to_cache),
    )
    episodes = dataset.get_col_data(train.data.ids.episode)
    steps = dataset.get_col_data(train.data.ids.step)
    _, inverse = np.unique(episodes, return_inverse=True)
    last = np.zeros(inverse.max() + 1, np.int64)
    np.maximum.at(last, inverse, steps)
    valid = np.nonzero(steps <= last[inverse] - cfg.goal_offset_steps)[0]
    num_eval = min(cfg.num_eval, len(valid))
    if num_eval < cfg.num_eval:
        print(f"only {num_eval} valid start steps, fewer than num_eval={cfg.num_eval}")
    rng = np.random.default_rng(cfg.seed)
    picked = np.sort(valid[rng.choice(len(valid), size=num_eval, replace=False)])

    frames = Frames(OmegaConf.to_container(train.data.frames, resolve=True))
    world = swm.World(
        ENV_ID,
        num_envs=num_eval,
        image_shape=(frames.spec["size"],) * 2,
        max_episode_steps=2 * cfg.eval_budget,
        frames=frames,
        spec=OmegaConf.to_container(train.data.eval),
        device=cfg.env_device,
        frames_device=cfg.render_device,
    )

    if cfg.weights == "random":
        policy = swm.policy.RandomPolicy(seed=cfg.seed)
    else:
        model = swm.wm.utils.load_pretrained(f"{cfg.run}/{cfg.weights}", cache_dir=cfg.paths.runs)
        model = model.to(cfg.solver.device).eval().requires_grad_(False)
        cost = swm.planning.ShootingCostEvaluator(model, hydra.utils.instantiate(cfg.objective))
        solver = hydra.utils.instantiate(cfg.solver, cost=cost)
        image = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            ]
        )
        # The training scalers: the action is planned in the space the model trained in.
        process = {"action": scalers(train.data, train.seed)["action"]}
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=plan, process=process, transform={"pixels": image, "goal": image}
        )
    world.set_policy(policy)

    start = time.perf_counter()
    metrics = world.evaluate(
        dataset=dataset,
        episodes_idx=episodes[picked].tolist(),
        start_steps=steps[picked].tolist(),
        goal_offset=cfg.goal_offset_steps,
        eval_budget=cfg.eval_budget,
    )
    seconds = time.perf_counter() - start
    world.close()

    print(f"success_rate {metrics['success_rate']} over {num_eval} episodes in {seconds:.0f} s")
    result = {
        "weights": cfg.weights,
        "seed": cfg.seed,
        "num_eval": num_eval,
        "success_rate": metrics["success_rate"],
        "episode_successes": metrics["episode_successes"].astype(int).tolist(),
        "eval_seconds": round(seconds, 1),
        "plan": dataclasses.asdict(plan),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    with (run_dir / "eval_results.jsonl").open("a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    run()
