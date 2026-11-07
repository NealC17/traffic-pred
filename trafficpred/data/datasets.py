import math
import os
import re
from dataclasses import dataclass, field

import numpy as np
from torch.utils.data import ConcatDataset

from trafficpred.data import files, mmnist
from trafficpred.data.windows import (FlowWindows, MovingMNISTGenerator, MovingMNISTSequences,
                                      base_datasets, limit)

SPLIT_KEYS = ("train_files", "val_files", "test_files", "train_size", "val_size", "test_size")


@dataclass
class Splits:
    dataset: str
    train: object
    val: object
    test: object
    lo: float
    hi: float
    channels: tuple
    t_in: int
    t_out: int
    split: dict = field(default_factory=dict)
    train_is_fixed: bool = True
    spatial_features: object = None

    @property
    def span(self):
        return self.hi - self.lo

    @property
    def shape(self):
        x, _ = self.train[0]
        return tuple(x.shape[1:])

    def _sample(self, samples):
        return np.linspace(0, len(self.train) - 1, min(len(self.train), samples)).astype(int)

    def train_mean(self, samples=2000):
        return float(np.mean([self.train[i][1].float().mean().item() for i in self._sample(samples)]))

    def input_stats(self, samples=2000):
        x = np.stack([self.train[i][0].numpy() for i in self._sample(samples)])
        return float(x.mean()), float(x.std())


class GridFlowData:
    def __init__(self, title, pattern, channels, steps_per_day, test_days, val_days, horizon=(4, 4),
                 spatial_file=None):
        self.title = title
        self.spatial_file = spatial_file
        self.pattern = re.compile(pattern)
        self.channels = channels
        self.test_frames = test_days * steps_per_day
        self.val_frames = val_days * steps_per_day
        self.horizon = horizon

    def files(self):
        return [n for n in files.available() if self.pattern.fullmatch(n)]

    def splits(self, name, t_in, t_out, train_files=None, val_files=None, test_files=None,
               train_size=None, val_size=None, test_size=None):
        names = self.files()
        if not names:
            raise FileNotFoundError(f"no {self.title} files under {files.ROOT}")
        if train_files or val_files or test_files:
            train, val, test = self._by_file(names, train_files, val_files, test_files)
        else:
            train, val, test = self._by_time(names)
        lo = min(float(a.min()) for a in train)
        hi = max(float(a.max()) for a in train)
        windows = lambda parts: ConcatDataset([FlowWindows(a, t_in, t_out, lo, hi) for a in parts])
        split = dict(train_files=train_files, val_files=val_files, test_files=test_files,
                     train_size=train_size, val_size=val_size, test_size=test_size)
        return Splits(name, limit(windows(train), train_size, "train"),
                      limit(windows(val), val_size, "val"), limit(windows(test), test_size, "test"),
                      lo, hi, self.channels, t_in, t_out, split, spatial_features=self._spatial())

    def _spatial(self):
        if self.spatial_file is None:
            return None
        f = np.asarray(files.load(self.spatial_file), dtype=np.float32)
        f = f.reshape(f.shape[0], -1).T
        return (f - f.mean(0)) / (f.std(0) + 1e-6)

    def _by_time(self, names):
        *head, last = names
        frames = files.load(last)
        cut_test = len(frames) - self.test_frames
        cut_val = cut_test - self.val_frames
        if cut_val <= 0:
            raise ValueError(f"{last} is too short for the val and test periods")
        train = [files.load(n) for n in head] + [frames[:cut_val]]
        return train, [frames[cut_val:cut_test]], [frames[cut_test:]]

    def _by_file(self, names, train_files, val_files, test_files):
        if not (train_files and val_files and test_files):
            raise ValueError("splitting by file needs train_files, val_files and test_files")
        groups = [self._match(names, specs) for specs in (train_files, val_files, test_files)]
        shared = (set(groups[0]) & set(groups[1])) | (set(groups[0]) & set(groups[2])) \
            | (set(groups[1]) & set(groups[2]))
        if shared:
            raise ValueError(f"file(s) in more than one split: {sorted(shared)}")
        return [[files.load(n) for n in group] for group in groups]

    def _match(self, names, specs):
        out = []
        for spec in specs:
            hits = [n for n in names if str(spec) in os.path.basename(n)]
            if len(hits) != 1:
                raise ValueError(f"'{spec}' matches {hits or 'no file'}; {self.title} files: {names}")
            out.append(hits[0])
        return out


class MovingMNISTData:
    title = "MovingMNIST"
    channels = ("Digits",)
    horizon = (10, 10)

    def splits(self, name, t_in, t_out, train_files=None, val_files=None, test_files=None,
               train_size=None, val_size=None, test_size=None):
        if (t_in, t_out) != self.horizon:
            raise ValueError("Moving MNIST sequences are fixed at 10 in -> 10 out")
        if train_files or val_files or test_files:
            raise ValueError("--*-files only apply to file-based datasets such as taxibj")
        mmnist.download()
        digits = mmnist.digits()
        train = MovingMNISTGenerator(digits, length=int(train_size or 10000))
        val = MovingMNISTGenerator(digits, length=int(val_size or 1000), seed=12345)
        test = limit(MovingMNISTSequences(files.load(mmnist.TEST_FILE)), test_size, "test")
        split = dict(train_size=train_size, val_size=val_size, test_size=test_size)
        return Splits(name, train, val, test, 0.0, 255.0, self.channels, t_in, t_out, split,
                      train_is_fixed=False)


class SyntheticData:
    title = "Synthetic"
    channels = ("Inflow", "Outflow")
    horizon = (4, 4)

    def splits(self, name, t_in, t_out, train_files=None, val_files=None, test_files=None,
               train_size=None, val_size=None, test_size=None, frames=1500, size=32, seed=0):
        if train_files or val_files or test_files:
            raise ValueError("--*-files only apply to file-based datasets such as taxibj")
        f = self._frames(frames, size, seed)
        a, b = int(0.7 * frames), int(0.85 * frames)
        lo, hi = float(f[:a].min()), float(f[:a].max())
        win = lambda part: FlowWindows(part, t_in, t_out, lo, hi)
        split = dict(train_size=train_size, val_size=val_size, test_size=test_size)
        return Splits(name, limit(win(f[:a]), train_size, "train"), limit(win(f[a:b]), val_size, "val"),
                      limit(win(f[b:]), test_size, "test"), lo, hi, self.channels, t_in, t_out, split)

    @staticmethod
    def _frames(n, size, seed):
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:size, 0:size]
        centers = rng.uniform(4, size - 4, size=(5, 2))
        velocity = rng.normal(0, 0.3, size=(5, 2))
        f = np.zeros((n, 2, size, size), dtype=np.float32)
        for t in range(n):
            centers = np.clip(centers + velocity, 2, size - 3)
            volume = 1.0 + 0.8 * math.sin(2 * math.pi * t / 48)
            for c in range(2):
                f[t, c] = volume * sum(np.exp(-((yy - cy - c) ** 2 + (xx - cx) ** 2) / 8)
                                       for cy, cx in centers)
        return f + rng.uniform(0, 0.02, f.shape).astype(np.float32)


DATASETS = {
    "taxibj": GridFlowData("TaxiBJ", r"TaxiBJ/TaxiBJ20\d\d(_\d+)?\.npy", ("Inflow", "Outflow"),
                           steps_per_day=48, test_days=28, val_days=28),
    "bikenyc": GridFlowData("BikeNYC", r"BikeNYC/flow_data\.npy", ("Inflow", "Outflow"),
                            steps_per_day=24, test_days=10, val_days=10,
                            spatial_file="BikeNYC/poi_data.npy"),
    "mmnist": MovingMNISTData(),
    "synthetic": SyntheticData(),
}


def get_splits(dataset="taxibj", t_in=None, t_out=None, **split):
    spec = DATASETS[dataset]
    t_in = t_in or spec.horizon[0]
    t_out = t_out or spec.horizon[1]
    split = {k: v for k, v in split.items() if k in SPLIT_KEYS}
    return spec.splits(dataset, t_in, t_out, **split)


def scale_of(ds):
    first = base_datasets(ds)[0]
    return first.lo, first.lo + first.span


def _size(text):
    return float(text) if any(c in text for c in ".eE") else int(text)


def add_data_args(ap):
    g = ap.add_argument_group(
        "data",
        "Default split: the last 4 weeks (taxibj) or 10 days (bikenyc) are test, the same "
        "length before that is val, the rest is train. Name files to split by file instead, "
        "e.g. --train-files 2013 --val-files 2014 --test-files 2015 2016. Sizes keep the "
        "first N windows (or a fraction, e.g. 0.5).")
    g.add_argument("--dataset", choices=list(DATASETS), default="taxibj")
    g.add_argument("--t-in", type=int, help="input frames (dataset default if omitted)")
    g.add_argument("--t-out", type=int, help="frames to predict (dataset default if omitted)")
    g.add_argument("--train-files", nargs="+", metavar="NAME")
    g.add_argument("--val-files", nargs="+", metavar="NAME")
    g.add_argument("--test-files", nargs="+", metavar="NAME")
    g.add_argument("--train-size", type=_size, metavar="N")
    g.add_argument("--val-size", type=_size, metavar="N")
    g.add_argument("--test-size", type=_size, metavar="N")
    return g


def splits_from_args(args):
    return get_splits(args.dataset, args.t_in, args.t_out,
                      **{k: getattr(args, k, None) for k in SPLIT_KEYS})
