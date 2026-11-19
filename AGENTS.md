# insight

Research code for exploring world-model loss formulations between latent (JEPA) and pixel prediction, and their effects on representation, training stability, sample efficiency and robustness. It depends on three pinned libraries and holds only what they don't provide.

## Layers

- **stable-envs**: batched simulators with explicit `State`, `render(state)` on the GPU.
- **stable-worldmodel (swm)**: dataset format and loading, LeWM + SIGReg, checkpoints, `World`, solvers, planning objectives.
- **stable-pretraining (spt)**: `Module`, `Manager`, `DataModule`, GPU transforms, monitoring callbacks.
- **insight**: toy envs, frame specs, nuisance overlays, model variants, losses, metrics, configs, entry scripts.

## Rules

- **Reuse before building.** Start from each library's canonical usage (its scripts, examples and docs at the pinned commit). Deviate only when a concrete requirement fails on that path, and name every deviation and its reason in the PR that introduces it.
- **No stored pixels.** A dataset is recorded state rows in swm's format plus a frame spec; frames are drawn on the training GPU by one `Frames` object that training, measurement and planning share.
- **One pipeline.** Every variant plugs into the same data build, `train.py`, `eval.py` and `measure.py` through Hydra groups (`data/`, `model/`, `loss/`, `nuisance/`, `experiment/`). A paper's settings become config overrides, not a separate stack.
- **Ports, not checkouts.** Baselines from other repos are ported into `insight/` with their license in `licenses/`, never run from their own code.
- **One change per arm.** Every experiment arm names what it is compared against and changes one thing from it.
- Keep machine-specific paths, hosts and devices in `configs/local/local.yaml` (gitignored, from `local.yaml.template`).

## Workflow

- Branch from `dev` as `<prefix>/<goal>` (`feature`, `fix`, `refactor`; 2–3 snake-case words). PRs target `dev`.
- Commit subjects are short, lowercase and imperative; one coherent change per commit.
- Run with `uv run`; add dependencies with `uv add`; keep `uv.lock` committed.
- Tests guard silent failures (alignment, masking, gradient routing, seeding, resume). Each test names the failure it would catch.
- Code on an unmerged PR is untrusted: its results are not reported as findings until the PR merges to `dev`.
