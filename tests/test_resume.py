import csv
import importlib.util
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import ModelCheckpoint
from toy import write_toy_tworoom

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_script", ROOT / "scripts" / "train.py")
train_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_script)

EPOCHS = 3


def fit(data, runs, run, device):
    overrides = [
        "data=tworoom",
        f"data.dataset.name={data}",
        "data.dataset.frameskip=2",
        f"paths.runs={runs}",
        f"run={run}",
        "num_workers=1",
        "loader.batch_size=4",
        "loader.pin_memory=false",
        "model.predictor.depth=1",
        "model.predictor.heads=2",
        "model.predictor.dim_head=8",
        "model.predictor.mlp_dim=32",
        "trainer.devices=1",
        f"trainer.max_epochs={EPOCHS}",
        "+trainer.limit_train_batches=3",
        "+trainer.limit_val_batches=1",
        "+trainer.log_every_n_steps=1",
    ]
    if device == "cpu":
        overrides += ["trainer.accelerator=cpu", "trainer.precision=32"]
    else:
        # GPU kernels may differ from run to run unless deterministic, which would mask the resume.
        overrides += ["trainer.accelerator=gpu", "+trainer.deterministic=true"]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        train_script.run(compose("train", overrides=overrides))
    return runs / "checkpoints" / run


def metrics(run_dir):
    """Logged training values by step, over all launches of the run."""
    values = {}
    for path in sorted(run_dir.glob("spt/runs/*/*/*/**/metrics.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["step"] and row.get("train/loss"):
                    values[int(row["step"])] = {
                        k: row[k] for k in ("train/loss", "train/pred_loss", "train/sigreg_loss")
                    }
    return values


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
        ),
    ],
)
def test_resume_after_a_kill_matches_an_uninterrupted_run(tmp_path, monkeypatch, device):
    """A run killed after epoch 0 and rerun matches an uninterrupted one in weights and per-step losses.

    A resume that drops the optimizer, scheduler or RNG state, or repeats an epoch, trains to other
    weights. On a GPU, the predictor's dropout also draws from the CUDA generator, which must be
    restored too.
    """
    monkeypatch.setenv("STABLEWM_HOME", str(tmp_path / "home"))
    data = tmp_path / "toy.h5"
    write_toy_tworoom(data)
    runs = tmp_path / "runs"

    whole = fit(data, runs, "whole", device)

    save = ModelCheckpoint.on_train_epoch_end
    epochs_run = []
    killed = False

    def save_then_die(self, trainer, pl_module):
        epochs_run.append(trainer.current_epoch)
        save(self, trainer, pl_module)
        if not killed:
            raise RuntimeError("killed")

    monkeypatch.setattr(ModelCheckpoint, "on_train_epoch_end", save_then_die)
    with pytest.raises(RuntimeError, match="killed"):
        fit(data, runs, "killed", device)
    killed = True
    resumed = fit(data, runs, "killed", device)

    assert epochs_run == [0, 1, 2]  # the rerun continues at epoch 1
    name = f"weights_epoch_{EPOCHS}.pt"
    expected, actual = (torch.load(d / name, map_location="cpu") for d in (whole, resumed))
    for key, tensor in expected.items():
        torch.testing.assert_close(actual[key], tensor, rtol=0, atol=0, msg=key)
    logged = metrics(whole)
    assert logged and metrics(resumed) == logged
