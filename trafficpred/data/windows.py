import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from trafficpred.data import external


class FlowWindows(Dataset):
    def __init__(self, frames, t_in, t_out, lo, hi, ext=None):
        self.frames = frames
        self.ext = ext
        self.t_in, self.t_out = t_in, t_out
        self.lo, self.span = lo, max(hi - lo, 1e-8)
        self.n = max(len(frames) - t_in - t_out + 1, 0)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        w = np.asarray(self.frames[i:i + self.t_in + self.t_out], dtype=np.float32)
        w = torch.from_numpy((w - self.lo) / self.span)
        if self.ext is None:
            return w[:self.t_in], w[self.t_in:]
        return w[:self.t_in], w[self.t_in:], torch.from_numpy(external.window(self.ext, i, self.t_in, self.t_out))


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
