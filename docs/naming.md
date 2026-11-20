# Naming

## Runs

A run is named `<model>_<data>[_<arm>]_ep<epochs>_s<seed>`, in lowercase, with parts joined by `_`. The name is both the run directory under `checkpoints/` and the W&B run id.

- `model`: the model or method, e.g. `lewm`, `motion`.
- `data`: the `data` config, e.g. `pusht`, `tworoom`.
- `arm`: what this run ablates against its experiment's baseline, in shorthand: a hyperparameter's short name followed by its value (`lr3e-4`, `bs64`, `wd0.05`), or a component change (`nobn`). A baseline run has no arm.
- `ep<epochs>`: the training length, `trainer.max_epochs`. Checkpoints within a run keep their own names (`weights_epoch_5.pt`).
- `s<seed>`: the seed.

Name only what the run is about. Hyperparameters tuned for a method, but not the subject of the run, stay out of the name; they are in the run's `config.yaml`. A motion model with a tuned learning rate is `motion_pusht_ep10_s0`. A learning-rate ablation of LeWM is `lewm_pusht_lr3e-4_ep10_s0`.

| Run | Meaning |
| --- | --- |
| `lewm_pusht_ep10_s1` | LeWM on Push-T, 10 epochs, seed 1 |
| `lewm_tworoom_nobn_ep10_s1` | LeWM on TwoRoom without the projector's BatchNorm |
| `lewm_pusht_lr3e-4_ep10_s0` | LeWM on Push-T at learning rate 3e-4 |

W&B never reuses a deleted run id, so a run that is retrained after its W&B run was deleted takes a new name.

## W&B groups

Each experiment is one W&B group (`+wandb.config.group=<experiment>`), named in the same shorthand, e.g. `lewm_repro`. Its runs are the baseline and the arms compared against it. A Flywheel Results node links the group page rather than each run. Runs kept on W&B back a result; debug and smoke runs are deleted.
