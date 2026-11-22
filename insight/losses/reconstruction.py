"""The reconstruction baseline of Sobal et al., arXiv 2211.10831 (section 2 and appendix A, eq. 17).

The rollout starts from the first frame's embedding alone, the predictor is unrolled on its own predictions
over the window's actions, and every predicted embedding is decoded against its frame, with the gradient
running back through the unroll. The latent terms are logged only. The decoder is a module of the loss
config, beside the model.
"""

import torch

from insight.losses.common import log_step, pixel_error


def reconstruction(self, batch, stage, *, history_size):
    """Pixel error of the open-loop rollout from the first frame, decoded at every later frame."""
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    first = self.model.encode({"pixels": batch["pixels"][:, :1]})["emb"]
    # The window's own embeddings, for the logged terms and the monitors only.
    with torch.no_grad():
        output = self.model.encode(batch)
    emb = output["emb"]
    # Passing the first embedding keeps the encoder in the graph; rollout detaches one it encodes itself.
    info = {"pixels": batch["pixels"][:, None, :1], "emb": first[:, None]}
    # The last action leaves the window.
    pred_emb = self.model.rollout(info, batch["action"][:, None, :-1])["predicted_emb"][:, 0, 1:]

    with torch.no_grad():
        output["pred_loss"] = (pred_emb - emb[:, 1:]).pow(2).mean()
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["pixel_loss"] = pixel_error(self, pred_emb, batch, 1)
    output["loss"] = output["pixel_loss"]
    return log_step(self, stage, output, emb, history_size, 1)
