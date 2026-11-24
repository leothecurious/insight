"""Representation metrics computed over encoded evaluation windows.

Each metric is ``fn(lat, **params) -> dict``. ``run`` evaluates a configured
list of them.

Sources:
  state_probe   Gonzalez Laiz & Schneider, arXiv 2610.37588 (App. D)
  rollout_probe Sobal et al., arXiv 2211.10831: the dot's position read from open-loop predictions
  rank_spectrum EpiWM worldmodel/analysis/probes.py (the-puzzler/epijepa@0d43616)
"""

import numpy as np
import torch
from sklearn.metrics import r2_score

from insight.metrics import probes


def _frames(lat, min_t=0):
    """Window and position of every valid frame at or after ``min_t``, and
    whether its window is on the train side."""
    valid = lat.valid.numpy().copy()
    valid[:, :min_t] = False
    n, t = np.nonzero(valid)
    return n, t, lat.train.numpy()[n]


def _split(train, max_train, max_test, seed):
    """Index arrays of train and test frames, subsampled to the given caps."""
    rng = np.random.default_rng(seed)
    out = []
    for side, cap in ((train, max_train), (~train, max_test)):
        idx = np.nonzero(side)[0]
        if cap is not None and len(idx) > cap:
            idx = np.sort(rng.choice(idx, cap, replace=False))
        out.append(idx)
    return out


def _selected(lat, groups):
    return [i for i, g in enumerate(lat.target_group) if groups is None or g in groups]


def _state_features(lat, n, t, columns=None):
    """True state at frames (n, t): every target, angles as (cos, sin)."""
    y = lat.target.numpy()[n, t]
    columns = range(y.shape[1]) if columns is None else columns
    return np.concatenate([probes.encode_target(y[:, j], lat.target_angle[j]) for j in columns], 1)


def state_probe(
    lat,
    groups=None,
    kinds=("linear", "mlp"),
    windows=(1,),
    max_train=20000,
    max_test=5000,
    seed=0,
):
    """Decode each target from the latent with each probe kind and window.

    Window w reads [z_{t-w+1}; ...; z_t]. All windows use the same frames
    (t >= max(windows) - 1), so they differ only in the input.
    """
    z = lat.z.numpy()
    n, t, side = _frames(lat, max(windows) - 1)
    tr, te = _split(side, max_train, max_test, seed)
    state = _state_features(lat, n, t)
    out = {}
    for j in _selected(lat, groups):
        angle = lat.target_angle[j]
        y = lat.target.numpy()[n, t, j]
        control = probes.control_mae(state[tr], y[tr], state[te], y[te], angle, seed)
        res = {"control_mae": control, "n_train": len(tr), "n_test": len(te)}
        for w in windows:
            X_tr, X_te = (probes.windowed(z, n[rows], t[rows], w) for rows in (tr, te))
            for kind in kinds:
                res[f"{kind}_w{w}"] = probes.probe_errors(
                    kind, X_tr, y[tr], X_te, y[te], angle, control, seed
                )
        out[lat.target_names[j]] = res
    return out


def rollout_probe(lat, max_train=None, max_test=None, seed=0, chunk=1 << 18):
    """Joint linear position readout shared across all open-loop timesteps.

    The readout is fit over chunks of ``chunk`` frames, so every train frame can be used.

    Dot coordinates are fractions of the image side length. The paper-style
    summary averages timestep Euclidean RMSEs, rather than pooling frames.
    ``r`` is the mean RMSE over that of a predictor that always says the centre.
    """
    if lat.predicted_z is None:
        raise ValueError("rollout_probe requires predicted_z from rollout extraction")
    if any(lat.target_names.count(name) != 1 for name in ("dot_x", "dot_y")):
        raise ValueError("rollout_probe requires unique dot_x and dot_y targets")
    columns = [lat.target_names.index(name) for name in ("dot_x", "dot_y")]
    n, t, side = _frames(lat)
    tr, te = _split(side, max_train, max_test, seed)
    if not len(tr) or not len(te):
        raise ValueError("rollout_probe requires valid train and test frames")
    predicted_z, target = lat.predicted_z.numpy(), lat.target.numpy()[:, :, columns]

    def train_chunks():
        for i in range(0, len(tr), chunk):
            rows = tr[i : i + chunk]
            yield predicted_z[n[rows], t[rows]], target[n[rows], t[rows]]

    probe = probes.StreamedLinear(train_chunks)
    y = target[n[te], t[te]]
    squared_error = ((probe.predict(predicted_z[n[te], t[te]]) - y) ** 2).sum(axis=1)
    center_squared_error = ((0.5 - y) ** 2).sum(axis=1)
    test_t = t[te]
    counts = [int((test_t == step).sum()) for step in range(lat.valid.shape[1])]

    def summarize(errors):
        per_step = [
            float(np.sqrt(errors[test_t == step].mean())) if count else None
            for step, count in enumerate(counts)
        ]
        return {
            "rmse_per_step": per_step,
            "mean_rmse": float(np.mean([value for value in per_step if value is not None])),
            "pooled_rmse": float(np.sqrt(errors.mean())),
        }

    model = summarize(squared_error)
    center = summarize(center_squared_error)
    return {
        "units": "coordinate side length",
        "n_train": len(tr),
        "n_test": len(te),
        "n_test_per_step": counts,
        **model,
        "center_control": center,
        "r": model["mean_rmse"] / center["mean_rmse"],
        "r_pooled": model["pooled_rmse"] / center["pooled_rmse"],
    }


def reverse_probe(
    lat,
    groups=("state",),
    kinds=("linear",),
    max_train=20000,
    max_test=5000,
    seed=0,
):
    """Share of latent variance the true state explains: variance-weighted
    R^2 of the latent regressed on the targets in ``groups``."""
    z = lat.z.numpy()
    n, t, side = _frames(lat)
    tr, te = _split(side, max_train, max_test, seed)
    X = _state_features(lat, n, t, _selected(lat, groups))
    y = z[n, t]
    out = {}
    for kind in kinds:
        m = probes.fit_probe(kind, X[tr], y[tr], seed)
        out[kind] = float(r2_score(y[te], m.predict(X[te]), multioutput="variance_weighted"))
    return out


def effective_rank(Z):
    """exp of the entropy of the normalized singular values of centered Z."""
    Z = torch.as_tensor(Z, dtype=torch.float64)
    sv = torch.linalg.svdvals(Z - Z.mean(0))
    total = sv.sum()
    if total == 0:
        return 0.0, sv
    p = sv / total
    return float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum())), sv


def rank_spectrum(lat, frame=None, top=32, max_frames=None, seed=0, validation_only=False):
    """Effective rank and PCA spectrum for a seeded sample of valid frames.

    Constant embeddings have rank and variance shares of zero.
    """
    valid = lat.valid.numpy()
    if validation_only:
        valid = valid & ~lat.train.numpy()[:, None]
    if frame is None:
        n, t = np.nonzero(valid)
    else:
        n = np.flatnonzero(valid[:, frame])
        t = np.full_like(n, frame)
    if max_frames is not None and len(n) > max_frames:
        keep = np.sort(np.random.default_rng(seed).choice(len(n), max_frames, replace=False))
        n, t = n[keep], t[keep]
    if not len(n):
        raise ValueError("rank_spectrum needs at least one valid frame")
    erank, sv = effective_rank(lat.z[n, t])
    variance = sv.square()
    ev = variance / variance.sum() if erank else variance
    return {
        "erank": erank,
        "top2_var": float(ev[:2].sum()),
        "spectrum": ev[:top].tolist(),
        "dim": int(lat.z.shape[-1]),
        "n": len(n),
    }


METRICS = {f.__name__: f for f in (state_probe, rollout_probe, reverse_probe, rank_spectrum)}


def run(lat, metrics):
    """Evaluate each configured metric; ``metrics`` is a list of ``{name, **params}``."""
    results = {}
    for entry in metrics:
        params = dict(entry)
        name = params.pop("name")
        print(f"metric {name}", flush=True)
        results[name] = METRICS[name](lat, **params)
    return results
