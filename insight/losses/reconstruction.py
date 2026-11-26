"""The reconstruction baseline of Sobal et al., arXiv 2211.10831 (section 2 and appendix A, eq. 17).

The rollout starts from the first frame's embedding alone, the predictor is unrolled on its own predictions
over the window's actions, and every predicted embedding is decoded against its frame, with the gradient
running back through the unroll. The latent terms are logged on validation only. The decoder is a module of
the loss config, beside the model.
"""

import torch

from insight.losses.common import log_step


def reconstruction(self, batch, stage, *, history_size):
    """Pixel error of the open-loop rollout from the first frame, decoded at every later frame."""
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    validating = stage == "validate"
    if validating:
        output = self.model.encode(batch)
        emb = output["emb"]
        first = emb[:, :1]
    else:
        # The objective needs only the first frame; it is also what the training-time monitors read.
        output = self.model.encode({"pixels": batch["pixels"][:, :1]})
        emb = first = output["emb"]
    # Passing the first embedding keeps the encoder in the graph; rollout detaches one it encodes itself.
    info = {"pixels": batch["pixels"][:, None, :1], "emb": first[:, None]}
    # The last action leaves the window.
    pred_emb = self.model.rollout(info, batch["action"][:, None, :-1])["predicted_emb"][:, 0, 1:]

    if validating:
        output["pred_loss"] = (pred_emb - emb[:, 1:]).pow(2).mean()
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    pred_pixels = self.decoder(pred_emb.flatten(0, 1)).float()
    output["pixel_loss"] = (
        (pred_pixels - batch["pixels"][:, 1:].flatten(0, 1).float()).pow(2).mean()
    )
    output["loss"] = output["pixel_loss"]
    return log_step(self, stage, output, emb)
