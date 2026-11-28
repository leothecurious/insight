<h1 align="center">insight</h1>

<p align="center"><i>A workbench for world-model objectives between latent prediction and pixel prediction.</i></p>

---

`insight` is for exploring loss formulations that bridge latent (JEPA) and pixel-level world modelling, and for measuring what each one does to the learned representation: what it encodes, how stably it trains, how much data it needs and how it holds up under nuisance. It provides a LeWM baseline and config groups for extending objectives, architectures and datasets over one data pipeline, training loop, metric suite and planner.

It is a thin research layer over three libraries:

| Library | Provides |
|---|---|
| [stable-envs](https://github.com/galilai-group/stable-envs) | batched simulators with explicit state; `render(state)` on the GPU |
| [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) | dataset format and loading, LeWM + SIGReg, checkpoints, planning |
| [stable-pretraining](https://github.com/galilai-group/stable-pretraining) | training loop, GPU transforms, monitoring callbacks |

The pipeline reads recorded state and action rows and draws frames from a shared *frame spec*. Training, measurement and planning use the same renderer and frame spec.

## Installation

Requires [uv](https://docs.astral.sh/uv/) and a CUDA GPU.

```bash
git clone https://github.com/leothecurious/insight
cd insight
uv sync
```

Library versions are pinned to exact commits in `pyproject.toml`; `uv.lock` fixes the rest.

## Data

Datasets and checkpoints live under `$STABLEWM_HOME` (swm's default, `~/.stable_worldmodel/`). The LeWM releases (PushT, TwoRoom) are on [HuggingFace](https://huggingface.co/collections/quentinll/lewm); decompress with `tar --zstd -xvf archive.tar.zst`. The moving dot is recorded locally, in seconds: `uv run scripts/collect.py name=dot1m_train sequences=1000000 seed=10`, then `dot1m_val` (seed 11, `first_episode=1000000`, 10,000 sequences), `dot1m_probe` (12, `1010000`, 300,000) and `dot1m_test` (13, `1310000`, 10,000). 3-dots is recorded the same way with `data=threedots` and names `threedots1m_*`.

For machine-specific dataset and checkpoint locations, copy `configs/local/local.yaml.template` to `configs/local/local.yaml` and set the paths there.

## Run an experiment

Give each training run a name. To resume from its full training checkpoint, repeat the training command with the same settings and add `resume=true`.

```bash
uv run scripts/train.py data=tworoom run=lewm_tworoom_s0 seed=0
uv run scripts/eval.py run=lewm_tworoom_s0 weights=lewm_tworoom_epoch_100.pt protocol=lewm_tworoom
uv run scripts/measure.py run=lewm_tworoom_s0 weights=lewm_tworoom_epoch_100.pt
```

The TwoRoom evaluation preset uses one context frame and leaves the checkpoint's training history intact. Set `num_eval=200` to match the episode count used in our TwoRoom reproduction.

Evaluation writes `eval_results.jsonl`. Measurement fits state probes on training episodes, scores them on validation episodes, and writes `measure_results.jsonl`. Use `latents_path` to reuse encodings from the same checkpoint and sampled windows. Use `weights=random` to measure an untrained model or evaluate a random-action policy.

For matched training runs, pass the same `+data.split.seed=0` to reserve the same episodes and use the same dataset and measurement window settings. Measurement uses the checkpoint's saved split. Select `embedding=backbone_cls` to read pre-projector CLS features. For example, measure effective rank on up to 4,000 held-out frames with `'metrics=[{name:rank_spectrum,validation_only:true,max_frames:4000}]'`.

## Layout

```
insight/     data and frames, losses, planning adapter, representation metrics
scripts/     Hydra entry points: train, eval, measure
configs/     one config root for every script
tests/       regression tests for silent failures
```

## License

MIT.
