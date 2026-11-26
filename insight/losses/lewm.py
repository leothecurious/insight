"""LeWM objective as a stable-pretraining forward function (stable-worldmodel ``scripts/train/lewm.py``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from insight.losses.common import log_step

if TYPE_CHECKING:
    from stable_pretraining import Module


def lewm(
    self: Module,
    batch: dict[str, Any],
    stage: str,
    *,
    history_size: int,
    num_preds: int,
    sigreg_weight: float,
) -> dict[str, Any]:
    """Predict the next embeddings from ``history_size`` frames and regularise the embeddings with SIGReg."""
    # NaN actions mark the start of an episode.
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb, act_emb = output["emb"], output["act_emb"]
    pred_emb = self.model.predict(emb[:, :history_size], act_emb[:, :history_size])

    output["pred_loss"] = (pred_emb - emb[:, num_preds:]).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + sigreg_weight * output["sigreg_loss"]
    return log_step(self, stage, output, emb)
