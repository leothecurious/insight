# Naming

## Runs

A run is named `<model>_<data>[_<arm>]_s<seed>`, in lowercase, with parts joined by `_`. The name is both the run directory under `checkpoints/` and the W&B run ID.

- `model`: the model or method, e.g. `lewm`, `motion`.
- `data`: the `data` config, e.g. `pusht`, `tworoom`.
- `arm`: what this run ablates against its experiment's baseline, in shorthand: a hyperparameter's short name followed by its value (`lr3e-4`, `bs64`, `wd0.05`), or a component change (`nobn`). A baseline run has no arm.
- `s<seed>`: the seed.

Record hyperparameters in the run's `config.yaml`. Include a hyperparameter in the arm only when the experiment varies it. For a training-length sweep, use `ep<epochs>` for `trainer.max_epochs`. Checkpoints keep their own names, such as `weights_epoch_5.pt`.

| Run | Meaning |
| --- | --- |
| `lewm_pusht_s1` | LeWM on Push-T, seed 1 |
| `lewm_tworoom_nobn_s1` | LeWM on TwoRoom without the projector's BatchNorm |
| `lewm_pusht_lr3e-4_s0` | Learning-rate sweep arm at 3e-4 |
| `lewm_pusht_ep10_s0` | Training-length sweep arm at 10 epochs |

W&B never reuses a deleted run id, so a run that is retrained after its W&B run was deleted takes a new name.

## W&B

| Level | Holds | Example |
| --- | --- | --- |
| project `insight` | every run, so any two can be compared | |
| group | one Flywheel Experiment, named by its slug (`+wandb.config.group=<slug>`) | `insight-pusht-lewm-training` |
| `config.series` | one arm of that Experiment: the run name without `_s<seed>`, set by `train.yaml` | `lewm_pusht` |
| run | one seed | `lewm_pusht_s1` |

To plot one curve per arm, filter to the experiment's group and group the line plot by `config.series`. Select **Mean** for aggregation. For a seed spread band, use **Full fidelity** point aggregation and select **Std Dev** for the range. See the [W&B line plot settings](https://github.com/wandb/docs/blob/main/models/app/features/panels/line-plot/sampling.mdx#configure-shading).

A Flywheel Result links its Experiment's group page. Tags label runs across experiments (`reproduction`), and a W&B Report collects several experiments on one page. `job_type` keeps W&B's meaning, the run's role (`train`).

Runs kept on W&B back a result; debug and smoke runs are deleted.
