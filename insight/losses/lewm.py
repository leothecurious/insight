"""LeWM objective as a stable-pretraining forward function (stable-worldmodel ``scripts/train/lewm.py``)."""

from insight.losses.common import encode_predict, log_step


def lewm(self, batch, stage, *, history_size, num_preds, sigreg_weight):
    """Predict the next embeddings from ``history_size`` frames and regularise the embeddings with SIGReg."""
    output, emb, pred_emb = encode_predict(self, batch, history_size, num_preds)

    output["pred_loss"] = (pred_emb - emb[:, num_preds:]).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + sigreg_weight * output["sigreg_loss"]
    return log_step(self, stage, output, emb, history_size, num_preds)
