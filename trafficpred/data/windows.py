import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from trafficpred.data import external


class FlowWindows(Dataset):
    """aux: (T, K, H, W) extra per-frame input maps (e.g. --od), scaled like the flow and appended to
    the input frames' channels; the targets stay the flow channels."""

    def __init__(self, frames, t_in, t_out, lo, hi, ext=None, aux=None):
        self.frames = frames
        self.ext = ext
        self.aux = aux
        self.t_in, self.t_out = t_in, t_out
        self.lo, self.span = lo, max(hi - lo, 1e-8)
        self.n = max(len(frames) - t_in - t_out + 1, 0)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        w = np.asarray(self.frames[i:i + self.t_in + self.t_out], dtype=np.float32)
        w = torch.from_numpy((w - self.lo) / self.span)
        x = w[:self.t_in]
        if self.aux is not None:
            x = torch.cat([x, self._aux(np.arange(i, i + self.t_in))], 1)
        if self.ext is None:
            return x, w[self.t_in:]
        return x, w[self.t_in:], torch.from_numpy(external.window(self.ext, i, self.t_in, self.t_out))

    def _aux(self, rows):
        return torch.from_numpy((np.asarray(self.aux[rows], dtype=np.float32) - self.lo) / self.span)


class PeriodicFlowWindows(FlowWindows):
    """Windows of frames[start:stop] whose input also holds, for every target frame, the frames one
    period earlier (e.g. the same time yesterday / last week), from anywhere earlier in `frames`.

    Input = [frames P_1 back for each target step, ..., frames P_k back ..., the t_in recent frames]:
    the most recent frame stays last. Windows without that much history are left out.
    """

    def __init__(self, frames, t_in, t_out, lo, hi, periods, start=0, stop=None, aux=None):
        super().__init__(frames, t_in, t_out, lo, hi, aux=aux)
        self.periods = tuple(periods)
        stop = len(frames) if stop is None else stop
        self.first = max(start, max(self.periods) - t_in)   # target frame i + t_in has P frames before it
        self.n = max(stop - t_in - t_out + 1 - self.first, 0)

    @property
    def extra_in(self):
        return len(self.periods) * self.t_out

    def __getitem__(self, j):
        i = self.first + j
        x, y = super().__getitem__(i)
        targets = np.arange(i + self.t_in, i + self.t_in + self.t_out)
        rows = np.concatenate([targets - p for p in self.periods])
        past = np.asarray(self.frames[rows], dtype=np.float32)
        past = torch.from_numpy((past - self.lo) / self.span)
        if self.aux is not None:
            past = torch.cat([past, self._aux(rows)], 1)
        return torch.cat([past, x]), y


class MovingMNISTSequences(Dataset):
    def __init__(self, arr, t_in=10):
        self.arr, self.t_in = arr, t_in

    def __len__(self):
        return self.arr.shape[1]

    def __getitem__(self, i):
        seq = np.asarray(self.arr[:, i], dtype=np.float32) / 255.0
        seq = torch.from_numpy(seq).unsqueeze(1)
        return seq[:self.t_in], seq[self.t_in:]


class MovingMNISTGenerator(Dataset):
    def __init__(self, digits, length=10000, frames=20, t_in=10, size=64, num_digits=2, seed=None):
        self.digits = digits
        self.length, self.frames, self.t_in = length, frames, t_in
        self.size, self.num_digits = size, num_digits
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng(None if self.seed is None else (self.seed, i))
        T, S, lim = self.frames, self.size, self.size - 28
        seq = np.zeros((T, S, S), dtype=np.float32)
        for _ in range(self.num_digits):
            digit = self.digits[rng.integers(len(self.digits))]
            theta = rng.uniform(0, 2 * np.pi)
            velocity = np.array([np.cos(theta), np.sin(theta)]) * 0.1 * lim
            pos = rng.uniform(0, lim, 2)
            for t in range(T):
                x, y = int(pos[0]), int(pos[1])
                np.maximum(seq[t, y:y + 28, x:x + 28], digit, out=seq[t, y:y + 28, x:x + 28])
                pos = pos + velocity
                for d in range(2):
                    if pos[d] < 0:
                        pos[d], velocity[d] = -pos[d], -velocity[d]
                    elif pos[d] > lim:
                        pos[d], velocity[d] = 2 * lim - pos[d], -velocity[d]
        seq = torch.from_numpy(seq).unsqueeze(1)
        return seq[:self.t_in], seq[self.t_in:]


def limit(ds, size, name):
    if size is None:
        return ds
    n = len(ds)
    k = int(round(size * n)) if isinstance(size, float) and 0 < size <= 1 else int(size)
    if k <= 0:
        raise ValueError(f"{name}_size={size} leaves no windows")
    if k > n:
        print(f"[data] {name}_size={size} but the {name} split only has {n} windows; using all")
        return ds
    return Subset(ds, range(k))


def base_datasets(ds):
    if isinstance(ds, ConcatDataset):
        return [b for d in ds.datasets for b in base_datasets(d)]
    if isinstance(ds, Subset):
        return base_datasets(ds.dataset)
    return [ds]
