"""The reconstruction objective of Sobal et al.'s RSSM, arXiv 2211.10831 code (vladisai/JEPA_SSL_NeurIPS_2022 at
03eb1730, rssm.py:240-290).

Every frame of the window is encoded. ``h`` starts as the first frame's embedding and is unrolled over the
window's actions, taking the previous step's ``z``: the first ``z`` is a prior sample, later ones are posterior
samples of the frame just seen. Each frame is decoded from a prior sample and ``h``. The loss adds the
KL(prior || posterior) of each step, its batch and dimension mean floored at ``free_nats``. Their samples
(``.sample()``, rssm.py:264) carry no gradient, so the heads learn only from KL terms above the floor;
``reparameterize`` draws them with ``rsample`` instead. The latent terms are logged on validation only.
"""

import torch
from torch.distributions import kl_divergence

from insight.losses.common import log_step


def reconstruction_rssm(self, batch, stage, *, history_size, free_nats, reparameterize=False):
    """Pixel error of every frame decoded from ``(z, h)``, plus the floored KL of each later step."""
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)
    emb, act_emb = output["emb"], output["act_emb"]
    n_frames = emb.shape[1]

    def draw(dist):
        return dist.rsample() if reparameterize else dist.sample()

    h = emb[:, 0]
    z = draw(self.model.prior(h))
    beliefs, states, kls = [h], [z], []
    for t in range(n_frames - 1):
        h = self.model.step(h, act_emb[:, t], z)
        prior = self.model.prior(h, self.model.min_var)
        posterior = self.model.posterior(emb[:, t + 1], h)
        kls.append(kl_divergence(prior, posterior).mean())
        beliefs.append(h)
        states.append(draw(prior))
        z = draw(posterior)
    beliefs = torch.stack(beliefs, 1)
    kls = torch.stack(kls)

    output["kl_loss"] = kls.detach().mean()
    if stage == "validate":
        output["pred_loss"] = (beliefs[:, 1:] - emb[:, 1:]).pow(2).mean()
        output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    latent = torch.cat([torch.stack(states, 1), beliefs], -1)
    pred_pixels = self.decoder(latent.flatten(0, 1)).float()
    output["pixel_loss"] = (pred_pixels - batch["pixels"].flatten(0, 1).float()).pow(2).mean()
    # Their loss sums the frame errors and the KLs and divides by the number of steps.
    output["loss"] = (n_frames * output["pixel_loss"] + kls.clamp(min=free_nats).sum()) / (
        n_frames - 1
    )
    return log_step(self, stage, output, emb)
