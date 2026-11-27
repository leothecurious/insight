"""Record rollouts of a toy environment (the moving dot, 3-dots) as swm state rows.

stable-envs steps the batch of environments and the episodes are stored in stable-worldmodel's HDF5
format (one dataset per column, plus ``ep_len`` and ``ep_offset``), which ``swm.data.load_dataset``
reads back. The frame spec of the dataset maps State fields to columns, so the columns written here are
the ones ``Frames`` reads. No pixels are stored.

The file is written with h5py, as stable-envs ``examples/lewm/make_dataset.py`` does, and not with
``HDF5Writer``: the writer chunks every column by single rows, and loading such a 90,000-sequence state
file takes 10.4 GB of RAM at peak, against 0.9 GB for contiguous columns.
"""

import h5py
import numpy as np
import stable_envs as snv
import torch


def collect(path, frames, policy, sequences, steps, seed, first_episode, chunk, device="cpu"):
    """Write ``sequences`` episodes of ``steps + 1`` rows to ``path``.

    Row ``t`` of an episode holds the State before action ``t``; the last row's action leaves no next
    row and is NaN. Episodes are numbered from ``first_episode`` in the ``frames.ids`` columns. Noise overlays
    are keyed by episode and step, so files that share a run seed need disjoint episode ids.
    """
    assert sequences % chunk == 0, "sequences must be a multiple of chunk"
    assert first_episode + sequences < 2**31, "episode ids are stored as int32"
    spec = frames.spec
    env = snv.make(spec["env"], num_envs=chunk, device=device, **(spec.get("kwargs") or {}))
    # The env draws its start states from default_rng(seed); the policy needs a stream of its own.
    rng = np.random.default_rng([seed, 1])
    length = steps + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "x") as f:
        f["ep_len"] = np.full(sequences, length, np.int32)
        f["ep_offset"] = np.arange(sequences, dtype=np.int64) * length
        for first in range(0, sequences, chunk):
            start = env.reset(seed=seed) if first == 0 else env.reset()
            actions = policy(chunk, steps, float(env.action_high), rng).to(env.device)
            states = torch.cat(
                [start.clone().unsqueeze(1), env.step(actions, frames=None)["states"]], 1
            )

            # Action t must be the move from row t to row t + 1.
            pos = snv.get_field(states, env.action_leaf)
            moved = (pos[:, :-1] + actions).clamp(env.MARGIN, 1 - env.MARGIN)
            assert torch.equal(pos[:, 1:], moved)

            columns = {
                name: x.reshape(chunk, length, -1).cpu().numpy()
                for name, x in frames.columns(states.reshape(-1)).items()
            }
            columns["action"] = np.concatenate(
                [actions.cpu().numpy(), np.full((chunk, 1, actions.shape[-1]), np.nan, np.float32)],
                1,
            )
            episode = first_episode + first + np.arange(chunk, dtype=np.int32)
            columns[frames.ids["episode"]] = np.repeat(episode[:, None], length, 1)
            columns[frames.ids["step"]] = np.tile(np.arange(length, dtype=np.int32), (chunk, 1))
            for name, x in columns.items():
                if name not in f:
                    f.create_dataset(name, shape=(sequences * length, *x.shape[2:]), dtype=x.dtype)
                f[name][first * length : (first + chunk) * length] = x.reshape(-1, *x.shape[2:])
    env.close()
