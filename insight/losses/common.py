"""The logging and monitor embeddings that the LeWM and reconstruction objectives share."""


def log_step(self, stage, output, emb):
    """Log the losses of ``output`` and set the embeddings the training-time monitors read."""
    prefix = {"fit": "train", "validate": "val"}.get(stage, stage)
    # Every frame's embedding, (B * T, D), for the training-time monitors.
    output["embedding"] = emb.flatten(0, 1)
    self.log_dict(
        {f"{prefix}/{k}": v.detach() for k, v in output.items() if "loss" in k},
        on_step=True,
        sync_dist=True,
    )
    return output
