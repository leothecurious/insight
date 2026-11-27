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
    output = reconstruction(module, batch, "fit")

    grad_first, grad_rest = torch.autograd.grad(output["loss"], [first, rest], allow_unused=True)
    assert grad_first.abs().sum() > 0
    assert grad_rest is None


def rssm_module():
    from stable_worldmodel.wm.lewm.module import Embedder, Predictor

    from insight.models.rssm import RSSMLeWM

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(12, D)

        def forward(self, pixels, interpolate_pos_encoding=True):
            return SimpleNamespace(last_hidden_state=self.linear(pixels.flatten(1))[:, None])

    model = RSSMLeWM(
        embed_dim=D,
        min_var=0.1,
        encoder=Encoder(),
        predictor=Predictor(
            num_frames=1, depth=1, heads=1, mlp_dim=8, input_dim=D, hidden_dim=D, dim_head=D
        ),
        action_encoder=Embedder(input_dim=2, emb_dim=D),
    )
    # The predictor's modulation layers start at zero, which blocks any gradient through its conditioning.
    for p in model.predictor.parameters():
        if not p.any():
            torch.nn.init.normal_(p, std=0.1)
    decoder = torch.nn.Sequential(torch.nn.Linear(2 * D, 12), torch.nn.Unflatten(1, (3, 2, 2)))
    return SimpleNamespace(
        model=model,
        decoder=decoder,
        sigreg=SIGReg(knots=5, num_proj=8),
        global_step=0,
        log=lambda *a, **k: None,
        log_dict=lambda *a, **k: None,
    )


def test_rssm_posterior_learns_only_from_the_kl_above_the_free_nats():
    """The posterior is trained by the KL alone because the samples carry no gradient; the floor
    removes the KL's gradient below it. Taking the KL's minimum instead of its maximum with the floor
    would train the posterior with no error."""
    from insight.losses.reconstruction_rssm import reconstruction_rssm

    def posterior_grad(free_nats):
        torch.manual_seed(0)
        module = rssm_module()
        batch = {"pixels": torch.randn(B, 4, 3, 2, 2), "action": torch.randn(B, 4, 2)}
        output = reconstruction_rssm(module, batch, "fit", free_nats=free_nats)
        output["loss"].backward()
        grads = [p.grad for p in module.model.posterior_mu.parameters()]
        return sum(g.abs().sum() for g in grads if g is not None)

    assert posterior_grad(free_nats=1e6) == 0
    assert posterior_grad(free_nats=0.0) > 0
