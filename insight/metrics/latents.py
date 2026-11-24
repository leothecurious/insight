"""Encode rendered evaluation windows once for reuse across probes."""

from dataclasses import dataclass

import torch

PRECISION = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass
class Latents:
    """Per-window latents and labels.

    z        (N, T, D)  encoded frames at the selected embedding layer
    action   (N, T, A)  normalized actions, action t follows frame t
    valid    (N, T)     real (unpadded) frames, always a prefix
    target   (N, T, M)  probe targets in dataset units
    episode  (N,)       dataset episode of each window
    train    (N,)       probe-train side of the episode split
    predicted_z (N, T, D) optional open-loop rollout from frame 0, including frame 0
    embedding selected encoder output, projector or backbone_cls
    """

    z: torch.Tensor
    action: torch.Tensor
    valid: torch.Tensor
    target: torch.Tensor
    episode: torch.Tensor
    train: torch.Tensor
    target_names: list
    target_angle: list
    target_group: list
    predicted_z: torch.Tensor | None = None
    embedding: str = "projector"

    def save(self, path):
        torch.save(vars(self), path)

    @classmethod
    def load(cls, path):
        return cls(**torch.load(path, weights_only=False))


@torch.inference_mode()
def extract(model, loader, transform, precision="fp32", rollout=False, *, embedding="projector"):
    """Draw and encode every frame of each window. Returns z, action and valid.

    ``transform`` is the dataset's ``gpu_transform``, which turns the state rows of a batch into
    the normalized frames the model trained on. ``embedding`` selects the projected embedding or
    the backbone CLS token. With ``rollout``, the model also predicts each window from its first frame
    and actions, and ``predicted_z`` holds the projected embeddings.
    """
    device = next(model.parameters()).device
    transform.to(device)
    dtype = PRECISION[precision]
    if embedding not in ("projector", "backbone_cls"):
        raise ValueError(f"unknown embedding layer: {embedding}")
    if rollout and embedding != "projector":
        raise ValueError("rollout extraction requires embedding=projector")
    out = {"z": [], "action": [], "valid": []}
    if rollout:
        assert model.predictor.num_frames == 1, "the rollout reads one context frame"
        out["predicted_z"] = []
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        pixels = transform(batch)["pixels"]
        with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
            if embedding == "backbone_cls":
                z = _encode_backbone_cls(model, pixels)
            else:
                z = model.encode({"pixels": pixels})["emb"]
            if rollout:
                # The last action leaves the window.
                info = {"pixels": pixels[:, None, :1]}
                predicted = model.rollout(info, batch["action"][:, None, :-1])
                out["predicted_z"].append(predicted["predicted_emb"][:, 0].float().cpu())
        out["z"].append(z.float().cpu())
        out["action"].append(batch["action"].cpu())
        out["valid"].append(batch["valid"].cpu())
    # Pop each list as it is joined so the chunks and the joined tensor are never all held at once.
    return {k: torch.cat(out.pop(k)) for k in list(out)}


def _encode_backbone_cls(model, pixels):
    """Read the CLS tensor LeWM passes to its projector during encoding."""
    captured = []

    def capture(_module, args):
        captured.append(args[0])

    with model.projector.register_forward_pre_hook(capture):
        model.encode({"pixels": pixels})
    return captured[0].unflatten(0, pixels.shape[:2])
