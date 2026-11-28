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


def lewm_sequence(
    self: Module,
    batch: dict[str, Any],
    stage: str,
    *,
    history_size: int,
    num_preds: int,
    sigreg_weight: float,
    rollout: bool,
) -> dict[str, Any]:
    """LeWM's loss over every transition of a window, from its own rollout or from the observed frames.

    With ``rollout``, the predictor is unrolled on its own predictions from the first frame's embedding over
    the window's actions, as in the reconstruction baseline, and each prediction is compared with the
    embedding of its frame, so the error of the last step reaches the first prediction and the encoder.
    Without it, each frame's embedding predicts the next one from a context of one frame, which is native
    LeWM's one-step loss on the same windows and updates. SIGReg is computed on every frame's embedding in
    both cases.
    """
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb = output["emb"]
    if emb.shape[1] != 1 + num_preds:
        raise ValueError("the window holds the first frame and one frame per prediction")
    if rollout:
        # Passing the first embedding keeps the encoder in the graph; rollout detaches one it encodes itself.
        info = {"pixels": batch["pixels"][:, None, :1], "emb": emb[:, None, :1]}
        # The last action leaves the window.
        pred_emb = self.model.rollout(
            info, batch["action"][:, None, :-1], history_size=history_size
        )
        pred_emb = pred_emb["predicted_emb"][:, 0, 1:]
    else:
        if history_size != 1:
            raise ValueError("the observed-frame loss predicts from one frame")
        b, t = emb.shape[:2]
        # Each transition becomes its own one-frame context, so no prediction sees earlier frames.
        pred_emb = self.model.predict(
            emb[:, :-1].reshape(b * (t - 1), 1, -1),
            output["act_emb"][:, :-1].reshape(b * (t - 1), 1, -1),
        ).reshape(b, t - 1, -1)

    output["pred_loss"] = (pred_emb - emb[:, 1:]).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + sigreg_weight * output["sigreg_loss"]
    return log_step(self, stage, output, emb)
