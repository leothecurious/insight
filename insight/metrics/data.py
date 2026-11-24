"""Evaluation windows: sampling, state rows and probe targets.

A window is ``(episode, start, n_frames)``: ``n_frames``
observations taken every ``frameskip`` rows from ``start``, with the actions between them, as in
training. Windows shorter than the requested length are padded and masked. A window holds state rows;
the dataset's ``gpu_transform`` draws its frames on the GPU.
"""

import numpy as np
import stable_pretraining as spt
import torch


def side(x):
    """Dataset and episodes of one side of a ``build`` split: a ``Subset`` of an episode split, or a whole dataset."""
    if isinstance(x, spt.data.Subset):
        return x.dataset, np.unique([x.dataset.clip_indices[i][0] for i in x.indices])
    return x, np.arange(len(x.lengths))


def sample_windows(ds, episodes, num_frames, min_frames, start, n, seed):
    """Pick windows of at most ``num_frames`` observations, one per episode.

    The window starts at the episode's first row (``episode_start``) or at a uniform start that
    leaves ``num_frames`` frames where the episode is long enough (``any_start``).
    """
    rng = np.random.default_rng(seed)
    fs = ds.frameskip
    eligible = episodes[np.asarray(ds.lengths)[episodes] // fs >= min_frames]
    picked = rng.choice(eligible, min(n, len(eligible)), replace=False)
    windows = []
    for ep in picked:
        length = int(ds.lengths[ep])
        s = 0
        if start == "any_start":
            s = int(rng.integers(0, max(0, length - num_frames * fs) + 1))
        else:
            assert start == "episode_start", start
        windows.append((int(ep), s, min((length - s) // fs, num_frames)))
    return windows


class Windows(torch.utils.data.Dataset):
    """State rows and actions of each window, padded to ``num_frames``.

    Padding repeats the last valid row and uses zero actions; ``valid`` marks the real frames.
    """

    def __init__(self, ds, windows, num_frames, columns):
        self.ds = ds
        self.windows = windows
        self.num_frames = num_frames
        self.columns = columns

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        ep, s, n = self.windows[i]
        item = self.ds.load_chunk([ep], [s], [s + n * self.ds.frameskip])[0]
        pad = self.num_frames - n
        out = {}
        for c in self.columns:
            x = torch.as_tensor(item[c])
            if pad:
                tail = (
                    x.new_zeros(pad, *x.shape[1:])
                    if c == "action"
                    else x[-1:].expand(pad, *x.shape[1:])
                )
                x = torch.cat([x, tail])
            out[c] = torch.nan_to_num(x.float(), 0.0) if c == "action" else x
        out["valid"] = torch.arange(self.num_frames) < n
        return out


def column_targets(ds, specs):
    """Per-row target values for the whole dataset, shape (rows, len(specs)).

    Each spec reads ``column[:, index]``.
    """
    cols, out = {}, []
    for spec in specs:
        if spec.column not in cols:
            cols[spec.column] = np.asarray(ds.get_col_data(spec.column), dtype=np.float64)
        col = cols[spec.column]
        x = col[:, spec.get("index", 0)] if col.ndim > 1 else col
        assert np.ptp(x) > 0, f"target {spec.name} is constant"
        out.append(x)
    return np.stack(out, 1)


def window_rows(ds, windows, num_frames):
    """Dataset row of each window frame, shape (N, num_frames); padded frames repeat the last row."""
    j = np.arange(num_frames)
    return np.stack(
        [ds.offsets[ep] + s + np.minimum(j, n - 1) * ds.frameskip for ep, s, n in windows]
    )


def window_targets(ds, windows, num_frames, specs):
    """Targets at each window frame, shape (N, num_frames, len(specs)).

    Padded frames repeat the last valid frame's target.
    """
    return column_targets(ds, specs)[window_rows(ds, windows, num_frames)]


def windows_loader(train, val, frame_columns, measure, w, seed):
    """Loader over windows of the training and validation sides, plus targets and split.

    ``train`` and ``val`` come from ``insight.data.build.build``; ``frame_columns`` are the columns
    the frames are drawn from and ``measure`` is the ``measure`` block of the data config. Windows of
    the training side come first, ``w.n_train`` of them, then ``w.n_val`` of the validation side.
    """
    sets, targets, episodes, is_train = [], [], [], []
    for x, count, on_train_side in ((train, w.n_train, True), (val, w.n_val, False)):
        ds, side_episodes = side(x)
        for c in measure.columns:
            dim = ds.get_dim(c.column)
            assert dim == c.dim, f"{c.column} has {dim} dims, expected {c.dim}"
        windows = sample_windows(ds, side_episodes, w.frames, w.min_frames, w.start, count, seed)
        dataset = Windows(ds, windows, w.frames, ["action", *frame_columns])
        # Frames come from load_chunk and targets from row indices; they must name the same rows.
        rows = window_rows(ds, windows, w.frames)
        probes = {0, len(windows) // 2, int(np.argmin([n for _, _, n in windows]))}
        for c in frame_columns:
            column = ds.get_col_data(c)
            for i in probes:
                np.testing.assert_array_equal(dataset[i][c].numpy(), column[rows[i]])
        sets.append(dataset)
        targets.append(window_targets(ds, windows, w.frames, measure.targets))
        episodes += [w_[0] for w_ in windows]
        is_train += [on_train_side] * len(windows)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(sets),
        batch_size=w.batch_size,
        num_workers=w.num_workers,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, np.concatenate(targets), np.array(episodes), np.array(is_train)
