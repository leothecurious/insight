"""LeWM with the stochastic state of Sobal et al.'s RSSM (vladisai/JEPA_SSL_NeurIPS_2022 at 03eb1730, rssm.py).

The deterministic state ``h`` is the LeWM predictor's output and starts as the first frame's embedding, as
their belief starts as the encoder output. A stochastic state ``z`` joins the action in the predictor's
conditioning. The prior reads ``h``; the posterior reads the next frame's embedding and ``h``. The heads and
the scale rule follow their code, including the ReLU on every head output.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from stable_worldmodel.wm.lewm import LeWM
from torch import nn
from torch.distributions import Normal


def head(n_in, n_out, layers):
    blocks = []
    for i in range(layers):
        blocks += [nn.Linear(n_in if i == 0 else n_out, n_out), nn.ReLU()]
    return nn.Sequential(*blocks)


class RSSMLeWM(LeWM):
    """``min_var`` is added to the scale of the prior and posterior while training (rssm.py:262, 268)."""

    def __init__(self, *, embed_dim, min_var, **kwargs):
        super().__init__(**kwargs)
        self.min_var = min_var
        self.z_proj = nn.Linear(embed_dim, embed_dim)
        self.prior_mu = head(embed_dim, embed_dim, 1)
        self.prior_scale = head(embed_dim, embed_dim, 1)
        self.posterior_mu = head(2 * embed_dim, embed_dim, 2)
        self.posterior_scale = head(2 * embed_dim, embed_dim, 2)

    def prior(self, h, min_var=0.0):
        """Normal over ``z`` given ``h``."""
        return Normal(self.prior_mu(h), F.softplus(self.prior_scale(h)) + min_var)

    def posterior(self, emb, h):
        """Normal over ``z`` given the frame's embedding and ``h``."""
        x = torch.cat([emb, h], dim=-1)
        return Normal(self.posterior_mu(x), F.softplus(self.posterior_scale(x)) + self.min_var)

    def step(self, h, act_emb, z):
        """The next ``h`` from the last ``h``, the action leaving it and ``z``; all ``(B, D)``."""
        return self.predict(h[:, None], (act_emb + self.z_proj(z))[:, None])[:, 0]

    def rollout(self, info, action_sequence, history_size=None):
        """``predicted_emb`` ``(B, S, 1 + T, D)``: the first frame's embedding, then ``h`` after each action.

        ``z`` is drawn from the prior at every step, as their ``predict_sequence`` does (rssm.py:65-95).
        """
        assert info["pixels"].size(2) == 1, "the rollout reads one context frame"
        B, S, T = action_sequence.shape[:3]
        first = self.encode({"pixels": info["pixels"][:, 0]})["emb"][:, 0]
        h = first[:, None].expand(B, S, -1).flatten(0, 1)
        act_emb = self.action_encoder(rearrange(action_sequence, "b s t a -> (b s) t a"))
        z = self.prior(h).sample()
        beliefs = [h]
        for t in range(T):
            h = self.step(h, act_emb[:, t], z)
            z = self.prior(h).sample()
            beliefs.append(h)
        info["predicted_emb"] = rearrange(torch.stack(beliefs, 1), "(b s) t d -> b s t d", b=B)
        return info
