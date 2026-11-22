<h1 align="center">insight</h1>

<p align="center"><i>A workbench for world-model objectives between latent prediction and pixel prediction.</i></p>

---

`insight` is for exploring loss formulations that bridge latent (JEPA) and pixel-level world modelling, and for measuring what each one does to the learned representation: what it encodes, how stably it trains, how much data it needs and how it holds up under nuisance. Objectives, architectures and datasets are swappable config groups over one data pipeline, training loop, metric suite and planner, so any two runs differ only in what you changed.

It is a thin research layer over three libraries:

| Library | Provides |
|---|---|
| [stable-envs](https://github.com/galilai-group/stable-envs) | batched simulators with explicit state; `render(state)` on the GPU |
| [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) | dataset format and loading, LeWM + SIGReg, checkpoints, planning |
| [stable-pretraining](https://github.com/galilai-group/stable-pretraining) | training loop, GPU transforms, monitoring callbacks |

No dataset stores pixels. Each is recorded state rows plus a *frame spec*, and frames are drawn on the training GPU, so training, measurement and planning see the same images by construction.

## Installation

Requires [uv](https://docs.astral.sh/uv/) and a CUDA GPU.

```bash
git clone https://github.com/leothecurious/insight
cd insight
uv sync
```

Library versions are pinned to exact commits in `pyproject.toml`; `uv.lock` fixes the rest.

## Data

Datasets and checkpoints live under `$STABLEWM_HOME` (swm's default, `~/.stable_worldmodel/`). The LeWM releases (PushT, TwoRoom) are on [HuggingFace](https://huggingface.co/collections/quentinll/lewm); decompress with `tar --zstd -xvf archive.tar.zst`. The moving dot is recorded locally, in seconds: `uv run scripts/collect.py name=dot1m_train sequences=1000000 seed=10`, then `dot1m_val`, `dot1m_probe` and `dot1m_test` from seeds and `first_episode` ranges of their own (see `configs/collect.yaml`).

## Layout

```
insight/     research code: envs, data (frames, nuisance), models, losses, planning, metrics
scripts/     Hydra entry points: train, eval, measure, collect
configs/     one config root for every script
tests/       regression tests for silent failures
```

## License

MIT.
