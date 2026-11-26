"""The logging and monitor embeddings that the LeWM and reconstruction objectives share."""


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
