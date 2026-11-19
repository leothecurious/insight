<h1 align="center">insight</h1>

<p align="center"><i>Probing the failure modes of JEPA world models: objectives, architectures and toy datasets on one pipeline.</i></p>

---

`insight` studies when joint-embedding predictive world models learn the wrong thing: nuisance features, memorised trajectories, collapsed dynamics. It compares objectives (LeWM, pixel twins, grounded hybrids), architectures (MotionJEPA, H-JEPA) and fast toy datasets (moving dot, PushT with nuisance tags) under one data pipeline, training loop, metric suite and planner.

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

Datasets and checkpoints live under `$STABLEWM_HOME` (swm's default, `~/.stable_worldmodel/`). The LeWM releases (PushT, TwoRoom) are on [HuggingFace](https://huggingface.co/collections/quentinll/lewm); decompress with `tar --zstd -xvf archive.tar.zst`.

## Layout

```
insight/     research code: envs, data (frames, nuisance), models, losses, planning, metrics
scripts/     Hydra entry points: train, eval, measure, collect
configs/     one config root for every script
tests/       regression tests for silent failures
licenses/    licenses of ported code
```

## Contributing

Read [AGENTS.md](AGENTS.md) first: it holds the design rules (reuse the libraries' canonical usage, one pipeline, one change per experiment arm) and the branch and commit conventions. Branch from `dev`, open PRs against `dev`, and name any deviation from library usage in the PR.

## License

MIT. Ported code keeps its original license in [`licenses/`](licenses/).
