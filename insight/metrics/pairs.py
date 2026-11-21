"""What a representation keys on: the paired-input cosine, and the colour probe.

A sample of frames is encoded twice with a corner tag (``insight.data.nuisance.CornerSquare``): once
with a colour of its own (``own``) and once with the colour of a partner frame (``swap``). Three
mean-centred cosines compare the two passes:

  same_tag      frame A against frame B stamped with A's colour (content differs)
  same_content  frame A against itself stamped with B's colour (tag differs)
  baseline      frame A against frame B, each with its own colour

An encoder that follows the scene reads high on same_content and near 0 on same_tag; one that follows
the tag reads the other way round. Each is read at the backbone (the CLS token, the read of Enigma's
Fig. 9) and at the projector output, which is what the planner sees.

The colour probe decodes the stamped colour from the latent and reports the error relative to chance.

Ported from the le-wm fork ``metrics/pairs.py``, after Enigma-Incorporated/The-Obsessed-Encoder@37819425
(``common/pair_metrics.py``, ``leworldmodel/additional_files/callbacks/pair_analysis.py``). Their frames
are freshly rendered; ours are drawn from dataset rows, which the encoder may have seen in training.
"""

import numpy as np
import torch
import torch.nn.functional as F
from stable_pretraining.data import dataset_stats
from torchvision.transforms.v2.functional import normalize

from insight.data.frames import Frames
from insight.data.nuisance import CornerSquare
from insight.metrics import probes, stimuli

REPS = ("backbone", "projection")


def cyclic_derangement(n, rng):
    """Partner of every position: one cycle over a random order, no fixed points."""
    order = rng.permutation(n)
    partner = np.empty(n, dtype=np.int64)
    partner[order] = np.roll(order, -1)
    return partner


def stamping(data, seed, tag_size, leaves=None):
    """The ``Frames`` of a resolved ``data`` config without its overlays, plus ``leaves``, and the tag to stamp.

    The tag has the config's own tag size when it has one, else ``tag_size``, and the run's ``seed``.
    """
    spec = dict(data["frames"])
    run_tag = spec.pop("nuisance", {}).get("tag", {})
    spec["leaves"] = {**spec["leaves"], **(leaves or {})}
    tag = CornerSquare(run_tag.get("size", tag_size), "episode", seed)
    return Frames(spec, data["ids"]), tag


def render(frames, columns, device, chunk=512):
    """``(N, H, W, 3)`` uint8 CPU frames of the ``(N, 1, D)`` state ``columns``, drawn ``chunk`` rows at a time."""
    n = len(next(iter(columns.values())))
    return torch.cat(
        [
            frames.draw({k: v[i : i + chunk].to(device) for k, v in columns.items()})[:, 0].cpu()
            for i in range(0, n, chunk)
        ]
    )


def sample_frames(frames, ds, n, seed, device):
    """One random row from each of ``n`` random episodes of ``ds``, drawn through ``frames``.

    Frames of one episode look alike, so two from the same episode would pair as different content.
    A dataset with fewer than ``n`` episodes gives one frame each.
    """
    rng = np.random.default_rng(seed)
    episodes = rng.choice(len(ds.lengths), min(n, len(ds.lengths)), replace=False)
    # A chunk holds whole frameskip blocks; its first row is the frame.
    starts = rng.integers(ds.lengths[episodes] - ds.frameskip + 1)
    chunks = ds.load_chunk(episodes, starts, starts + ds.frameskip)
    columns = {
        column: torch.stack([torch.as_tensor(chunk[column][:1]) for chunk in chunks])
        for column in {leaf["column"] for leaf in frames.spec["leaves"].values()}
    }
    return render(frames, columns, device)


@torch.inference_mode()
def encode(model, frames, batch_size=128):
    """Backbone (CLS token) and projector output of ``(N, H, W, 3)`` uint8 frames, on CPU.

    Frames are scaled and normalized as ``build`` does.
    """
    device = next(model.parameters()).device
    out = {rep: [] for rep in REPS}
    for i in range(0, len(frames), batch_size):
        pixels = frames[i : i + batch_size].to(device).permute(0, 3, 1, 2).float().div(255)
        pixels = normalize(pixels, **dataset_stats.ImageNet)
        cls = model.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 0]
        out["backbone"].append(cls.float().cpu())
        out["projection"].append(model.projector(cls).float().cpu())
    return {rep: torch.cat(v) for rep, v in out.items()}


def pair_cosines(own, swap, partner):
    """Mean cosine of each pairing, centred on the mean of the ``own`` pass."""
    n = len(partner)
    identity = np.arange(n)
    inverse = np.empty(n, dtype=np.int64)
    inverse[partner] = identity
    # swap row k carries the colour of partner(k), so colour k sits on swap row inverse[k]
    pairs = {
        "same_tag": (own, identity, swap, inverse),
        "same_content": (own, identity, swap, identity),
        "baseline": (own, identity, own, partner),
    }
    mu = own.mean(0)
    return {
        name: float(F.cosine_similarity(a[ai] - mu, b[bi] - mu, dim=1).mean())
        for name, (a, ai, b, bi) in pairs.items()
    }


def stamped(frames, tag, ids):
    """Copy of the ``(N, H, W, 3)`` ``frames`` with the colours that ``tag`` draws for the episode ``ids``."""
    return tag(frames.clone(), torch.as_tensor(ids))


def colour_probe(z, colours, kinds=("ridge", "mlp"), train_fraction=0.8, seed=0):
    """Decode the stamped colour from latents ``z`` (n, d); mean of the r, g, b errors.

    Colours are in 0-255. The error ratio eps compares each probe with an MLP from the true
    colour to shuffled colours (see ``probes``): 0 is exact, 100 chance.
    """
    colours = colours.numpy().astype(np.float64)
    z = z.numpy()
    order = np.random.default_rng(seed).permutation(len(z))
    tr, te = np.split(order, [round(train_fraction * len(z))])
    controls = [
        probes.control_mae(colours[tr], colours[tr, c], colours[te], colours[te, c], False, seed)
        for c in range(3)
    ]
    out = {}
    for kind in kinds:
        res = [
            probes.probe_errors(
                kind, z[tr], colours[tr, c], z[te], colours[te, c], False, controls[c], seed
            )
            for c in range(3)
        ]
        out[kind] = {k: float(np.mean([r[k] for r in res])) for k in ("mae", "eps", "r2")}
    return out


def measure_colour(model, frames, tag, kinds=("ridge", "mlp"), train_fraction=0.8, seed=0):
    """Colour probe of ``model`` on ``frames``, each stamped with a colour of its own from ``tag``."""
    ids = np.arange(len(frames))
    own = encode(model, stamped(frames, tag, ids))
    colours = tag.colour(torch.as_tensor(ids))
    return {rep: colour_probe(own[rep], colours, kinds, train_fraction, seed) for rep in REPS}


def measure(model, frames, tag, seed=0):
    """Pair cosines of ``model`` on ``frames`` stamped by ``tag``."""
    n = len(frames)
    partner = cyclic_derangement(n, np.random.default_rng(seed))
    own = encode(model, stamped(frames, tag, np.arange(n)))
    swap = encode(model, stamped(frames, tag, partner))
    return {rep: pair_cosines(own[rep], swap[rep], partner) for rep in REPS}


def measure_t_position(model, frames, n, tag, device, seed=0):
    """Pair cosines with the goal outline as the tag and the block pose as the content.

    The t_position suite of Enigma-Incorporated/The-Obsessed-Encoder@37819425. ``n`` scenes carry a
    block pose and a goal pose; the ``swap`` pass gives each scene its partner's goal. ``frames`` draws
    the goal outline (``stimuli.GOAL_LEAVES``). Every frame carries the colour that ``tag`` draws for
    episode 0, so the corner colour is held fixed across the suite.
    """
    rng = np.random.default_rng(seed)
    state, goal = stimuli.sample_orig(rng, n), stimuli.sample_dest(rng, n)
    partner = cyclic_derangement(n, rng)

    def encode_scenes(goal_pose):
        scenes = render(frames, {"state": state, "goal_pose": goal_pose}, device)
        return encode(model, stamped(scenes, tag, np.zeros(n, dtype=np.int64)))

    own, swap = encode_scenes(goal), encode_scenes(goal[partner])
    return {rep: pair_cosines(own[rep], swap[rep], partner) for rep in REPS}
