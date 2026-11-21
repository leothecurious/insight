from types import SimpleNamespace

import pytest
import torch
from stable_worldmodel.wm.loss import SIGReg

from insight.losses.common import grad_shares
from insight.losses.pixel import grounded, latent_twin, pixel

B, T, D = 6, 2, 4


def stub(emb):
    """What a loss reads from ``spt.Module``, with embeddings supplied by the test so that their gradients can be read."""
    predictor = torch.nn.Linear(D, D)
    model = SimpleNamespace(
        encode=lambda info: {**info, "emb": emb, "act_emb": info["action"]},
        predict=lambda emb, act_emb: predictor(emb + act_emb),
    )
    decoder = torch.nn.Sequential(torch.nn.Linear(D, 12), torch.nn.Unflatten(1, (3, 2, 2)))
    return SimpleNamespace(
        model=model,
        decoder=decoder,
        sigreg=SIGReg(knots=5, num_proj=8),
        global_step=0,
        log=lambda *a, **k: None,
        log_dict=lambda *a, **k: None,
    )


@pytest.mark.parametrize(
    ("loss", "weights", "target_trained", "sigreg_trained"),
    [
        (pixel, {}, False, False),
        (grounded, {"beta": 1.0, "sigreg_weight": 0.09, "grad_share_every": 1}, True, True),
        (latent_twin, {"lam": 1.0, "grad_share_every": 1}, False, False),
    ],
)
def test_pixel_losses_route_gradients_as_specified(loss, weights, target_trained, sigreg_trained):
    """The pixel twin trains on pixels only, the latent twin's target is a stop-gradient and grounded
    trains the latent term and SIGReg through the target frame as well. A leak would let the encoder
    chase its own target or be regularised by a term the objective omits, and the logged losses would
    look the same."""
    emb = torch.randn(B, T, D, requires_grad=True)
    batch = {"pixels": torch.randn(B, T, 3, 2, 2), "action": torch.randn(B, T, D)}
    module = stub(emb)
    output = loss(module, batch, "fit", history_size=1, num_preds=1, **weights)

    grads = torch.autograd.grad(output["loss"], [emb, *module.decoder.parameters()])
    assert grads[0][:, 0].abs().sum() > 0  # the context frame trains through the prediction
    assert (grads[0][:, 1].abs().sum() > 0) == target_trained
    assert all(g.abs().sum() > 0 for g in grads[1:])  # every objective trains the decoder
    assert output["sigreg_loss"].requires_grad == sigreg_trained


def test_grad_shares_split_the_gradient_norm_at_the_embedding():
    emb = torch.randn(3, 4, requires_grad=True)
    shares = grad_shares({"a": (2 * emb).sum(), "b": (6 * emb).sum()}, emb)
    assert float(shares["a"]) == pytest.approx(0.25) and float(shares["b"]) == pytest.approx(0.75)
    assert emb.grad is None
