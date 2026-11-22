"""Pieces the objectives share: encode and predict, the pixel error and logging.

Ported from le-wm PR #8 (``lewm_forward`` in ``train.py``).
"""

import torch
import torch.nn.functional as F


def encode_predict(self, batch, history_size, num_preds):
    """Embed the frames and predict the next embeddings from ``history_size`` of them.

    Returns the encode output (a dict with ``emb``), the embeddings ``(B, T, D)`` and the predictions.
    The target of prediction ``i`` is ``emb[:, num_preds + i]``.
    """
    # NaN actions mark the start of an episode.
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)
    emb, act_emb = output["emb"], output["act_emb"]
    pred_emb = self.model.predict(emb[:, :history_size], act_emb[:, :history_size])
    return output, emb, pred_emb


def pixel_error(self, pred_emb, batch, num_preds):
    """Mean squared error between the decoded predictions and the next frames, area-resized to the decoder's output."""
    pred_pixels = self.decoder(pred_emb.flatten(0, 1)).float()
    target = batch["pixels"][:, num_preds:].flatten(0, 1).float()
    target = F.interpolate(target, size=pred_pixels.shape[-2:], mode="area")
    return (pred_pixels - target).pow(2).mean()


def log_step(self, stage, output, emb, history_size, num_preds):
    """Log the losses of ``output`` and set the embeddings the training-time monitors read.

    On validation it adds ``copy_loss``, the error of copying each context embedding forward.
    """
    prefix = {"fit": "train", "validate": "val"}.get(stage, stage)
    if stage == "validate":
        output["copy_loss"] = (emb[:, :history_size] - emb[:, num_preds:]).detach().pow(2).mean()
    # Every frame's embedding, (B * T, D), for the training-time monitors.
    output["embedding"] = emb.flatten(0, 1)
    self.log_dict(
        {f"{prefix}/{k}": v.detach() for k, v in output.items() if "loss" in k},
        on_step=True,
        sync_dist=True,
    )
    return output
