"""Objectives that predict the next frame's pixels, ported from le-wm PR #8 (``lewm_forward`` in train.py).

``pixel`` is the pixel twin: the decoded prediction's pixel error alone. ``grounded`` adds it to LeWM's
objective with weight ``beta``. ``latent_twin`` adds a stop-gradient latent error to it with weight
``lam``. Every objective logs the latent terms; the ones that do not train on a term compute it under
``no_grad``. The decoder is a module of the loss config, beside the model.
"""

import torch

from insight.losses.common import encode_predict, log_step, pixel_error


def pixel(self, batch, stage, *, history_size, num_preds):
    """Pixel error of the decoded prediction; the latent terms are logged only."""
    output, emb, pred_emb = encode_predict(self, batch, history_size, num_preds)

    with torch.no_grad():
        output["pred_loss"] = (pred_emb - emb[:, num_preds:]).pow(2).mean()
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["pixel_loss"] = pixel_error(self, pred_emb, batch, num_preds)
    output["loss"] = output["pixel_loss"]
    return log_step(self, stage, output, emb, history_size, num_preds)


def grounded(self, batch, stage, *, history_size, num_preds, beta, sigreg_weight, grad_share_every):
    """LeWM's objective plus ``beta`` times the pixel error."""
    output, emb, pred_emb = encode_predict(self, batch, history_size, num_preds)

    output["pred_loss"] = (pred_emb - emb[:, num_preds:]).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["pixel_loss"] = pixel_error(self, pred_emb, batch, num_preds)
    terms = {"latent": output["pred_loss"], "pixel": beta * output["pixel_loss"]}
    output["loss"] = sum(terms.values()) + sigreg_weight * output["sigreg_loss"]
    return log_step(self, stage, output, emb, history_size, num_preds, terms, grad_share_every)


def latent_twin(self, batch, stage, *, history_size, num_preds, lam, grad_share_every):
    """Pixel error plus ``lam`` times the prediction error against a stop-gradient target; SIGReg is logged only."""
    output, emb, pred_emb = encode_predict(self, batch, history_size, num_preds)

    output["pred_loss"] = (pred_emb - emb[:, num_preds:].detach()).pow(2).mean()
    with torch.no_grad():
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["pixel_loss"] = pixel_error(self, pred_emb, batch, num_preds)
    terms = {"latent": lam * output["pred_loss"], "pixel": output["pixel_loss"]}
    output["loss"] = sum(terms.values())
    return log_step(self, stage, output, emb, history_size, num_preds, terms, grad_share_every)
