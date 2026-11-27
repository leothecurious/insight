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


def rssm_module(zero_init_z=False):
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
        zero_init_z=zero_init_z,
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


def rssm_posterior_grad(free_nats, kl_floor_over="dims_mean"):
    from insight.losses.reconstruction_rssm import reconstruction_rssm

    torch.manual_seed(0)
    module = rssm_module()
    batch = {"pixels": torch.randn(B, 4, 3, 2, 2), "action": torch.randn(B, 4, 2)}
    output = reconstruction_rssm(
        module, batch, "fit", free_nats=free_nats, kl_floor_over=kl_floor_over
    )
    output["loss"].backward()
    grads = [p.grad for p in module.model.posterior_mu.parameters()]
    return torch.cat([g.flatten() for g in grads]), output["kl_loss"]


def test_rssm_reconstruction_reaches_the_posterior_heads():
    """A floor above the KL removes its gradient. With ``.sample()`` the posterior would then get none
    at all and stay at its initialization."""
    grad, _ = rssm_posterior_grad(free_nats=1e6)
    assert grad.abs().sum() > 0


def test_rssm_floor_over_dims_sum_leaves_the_floor_below_the_summed_kl():
    """The KL's own gradient is the change in the heads' gradient when the floor moves from above it to
    ``free_nats``. A floor between the per-dimension mean and the sum over dimensions must pass the sum's
    gradient and block the mean's; reading the mean for ``dims_sum`` would keep the heads untrained."""
    baseline, _ = rssm_posterior_grad(free_nats=1e6)
    _, kl_mean = rssm_posterior_grad(free_nats=0.0)
    floor = 2 * kl_mean.item()

    def kl_grad(kl_floor_over):
        grad, _ = rssm_posterior_grad(free_nats=floor, kl_floor_over=kl_floor_over)
        return (grad - baseline).abs().sum()

    assert kl_grad("dims_mean") == 0
    assert kl_grad("dims_sum") > 0


def test_rssm_zero_init_z_adds_nothing_to_the_conditioning():
    """``zero_init_z`` must reach ``z_proj``; otherwise ``z`` enters the predictor as untrained noise."""
    module = rssm_module(zero_init_z=True)
    assert module.model.z_proj(torch.randn(B, D)).abs().sum() == 0
