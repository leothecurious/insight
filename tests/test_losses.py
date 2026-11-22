from types import SimpleNamespace

import torch
from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.loss import SIGReg

from insight.losses.reconstruction import reconstruction

B, T, D = 6, 5, 4


def test_reconstruction_trains_the_encoder_through_the_rollout_of_the_first_frame():
    """Only the first frame's embedding enters the rollout, and it stays in the graph. LeWM's rollout
    detaches an embedding it encodes itself, which would leave the encoder untrained, and a rollout fed
    later embeddings would be teacher-forced; both would train without error."""
    first = torch.randn(B, 1, D, requires_grad=True)
    rest = torch.randn(B, T - 1, D, requires_grad=True)
    batch = {"pixels": torch.randn(B, T, 3, 2, 2), "action": torch.randn(B, T, D)}
    predictor = torch.nn.Linear(D, D)
    model = SimpleNamespace(
        encode=lambda info: {
            **info,
            "emb": first if info["pixels"].shape[1] == 1 else torch.cat([first, rest], 1),
            "act_emb": info.get("action"),
        },
        predict=lambda emb, act_emb: predictor(emb + act_emb),
        predictor=SimpleNamespace(num_frames=1),
        action_encoder=torch.nn.Identity(),
    )
    model.rollout = lambda info, actions: LeWM.rollout(model, info, actions)
    module = SimpleNamespace(
        model=model,
        decoder=torch.nn.Sequential(torch.nn.Linear(D, 12), torch.nn.Unflatten(1, (3, 2, 2))),
        sigreg=SIGReg(knots=5, num_proj=8),
        log_dict=lambda *a, **k: None,
    )
    output = reconstruction(module, batch, "fit", history_size=1)

    grad_first, grad_rest = torch.autograd.grad(output["loss"], [first, rest], allow_unused=True)
    assert grad_first.abs().sum() > 0
    assert grad_rest is None
