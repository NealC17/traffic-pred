import math
import os
import re
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from trafficpred.data import external as ext_data
from trafficpred.data import files, mmnist
from trafficpred.data.windows import (FlowWindows, MovingMNISTGenerator, MovingMNISTSequences,
                                      PeriodicFlowWindows, base_datasets, limit)

SPLIT_KEYS = ("train_files", "val_files", "test_files", "train_size", "val_size", "test_size", "external",
              "periods", "period_shift", "test_days", "od")


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
    ext_dim: int = 0
    extra_in: int = 0   # input frames besides the t_in recent ones (e.g. --periods day week)
    extra_lags: tuple = ()   # how far back each of those extra frames is, in input order
    trend_lags: frozenset = frozenset()   # the lags that belong to a weekly period (ST-ResNet's trend)
    aux_in: int = 0   # extra input channels per frame after the flow ones (--od: 2), not predicted

    @property
    def in_frames(self):
        return self.t_in + self.extra_in

    @property
    def span(self):
        return self.hi - self.lo

    @property
    def shape(self):
        """(C, H, W) of the flow: the predicted channels (inputs carry aux_in more)."""
        x = self.train[0][0]
        return (x.shape[1] - self.aux_in, *x.shape[2:])

    def _sample(self, samples):
        return np.linspace(0, len(self.train) - 1, min(len(self.train), samples)).astype(int)

    def train_mean(self, samples=2000):
        return float(np.mean([self.train[i][1].float().mean().item() for i in self._sample(samples)]))

    def input_stats(self, samples=2000):
        x = np.stack([self.train[i][0].numpy()[:, :self.shape[0]] for i in self._sample(samples)])
        return float(x.mean()), float(x.std())


class GridFlowData:
    def __init__(self, title, pattern, channels, steps_per_day, test_days, val_days, horizon=(4, 4),
                 spatial_file=None, has_external=False, source=None, calendar=None, od_file=None):
        self.title = title
        self.od_file = od_file   # per-frame origin-destination maps (tools/import_stdn_od.py), for --od
        self.source = source   # where to get the files, shown when they are missing
        self.calendar = calendar   # (start date, holidays): calendar-only external features
        self.has_external = has_external or calendar is not None
        self.spatial_file = spatial_file
        self.pattern = re.compile(pattern)
        self.channels = channels
        self.steps_per_day = steps_per_day
        self.test_frames = test_days * steps_per_day
        self.val_frames = val_days * steps_per_day
        self.horizon = horizon

    def files(self):
        return [n for n in files.available() if self.pattern.fullmatch(n)]

    def period_steps(self, periods, shift=0):
        """'day' / 'week' / 'Nday' / 'Nweek' / a number of steps -> (steps largest first, the weekly ones).

        shift Q also takes the Q steps on either side of each period (STDN's periodically shifted
        history: the same time on earlier days drifts by a step or two). Weekly = a whole number of
        weeks before shifting; those are ST-ResNet's trend frames, the rest its period frames."""
        week = 7 * self.steps_per_day
        base = []
        for p in periods:
            m = re.fullmatch(r"(\d*)(day|week)", str(p))
            step = (int(m.group(1) or 1) * (self.steps_per_day if m.group(2) == "day" else week) if m
                    else int(p) if str(p).lstrip("-").isdigit() else None)
            if step is None or step <= 0:
                raise ValueError(f"period {p!r} must be a positive number of steps, or day / week / Nday / Nweek")
            base.append(step)
        if shift < 0 or (shift and shift >= min(base)):
            raise ValueError(f"--period-shift {shift} must be >= 0 and smaller than the shortest period")
        steps, trend = set(), set()
        for b in base:
            near = range(b - shift, b + shift + 1)
            steps.update(near)
            if b % week == 0:
                trend.update(near)
        return sorted(steps, reverse=True), frozenset(trend)

    def splits(self, name, t_in, t_out, train_files=None, val_files=None, test_files=None,
               train_size=None, val_size=None, test_size=None, external=False, periods=None,
               period_shift=0, test_days=None, od=False):
        names = self.files()
        if not names:
            hint = f"; {self.source}" if self.source else ""
            raise FileNotFoundError(f"no {self.title} files under {files.ROOT}{hint}")
        if external and not self.has_external:
            raise ValueError(f"{self.title} has no external data (weather / calendar); drop --external")
        if train_files or val_files or test_files:
            parts = self._by_file(names, train_files, val_files, test_files)
        else:
            parts = self._by_time(names, test_days)
        frames = [[files.load(n)[rows] for n, rows in group] for group in parts]
        lo = min(float(a.min()) for a in frames[0])
        hi = max(float(a.max()) for a in frames[0])
        ext = [[None] * len(group) for group in parts]
        if external and self.calendar:
            start, holidays = self.calendar
            feats = {n: ext_data.calendar(start, len(files.load(n)), self.steps_per_day, holidays)
                     for group in parts for n, _ in group}
            ext = [[feats[n][rows] for n, rows in group] for group in parts]
        elif external:
            raw = {n: ext_data.raw_for(n, len(files.load(n))) for group in parts for n, _ in group}
            mean, std = ext_data.weather_stats([raw[n][rows] for n, rows in parts[0]])
            ext = [[ext_data.features(raw[n], mean, std)[rows] for n, rows in group] for group in parts]
        aux = [[None] * len(group) for group in parts]
        if od:
            if self.od_file is None:
                raise ValueError(f"{self.title} has no origin-destination data; drop --od")
            if train_files or val_files or test_files or test_days or len(names) > 1:
                raise ValueError("--od goes with the default time split only (its kernel is fitted on that train split)")
            od_maps = files.load(self.od_file)
            if len(od_maps) != len(files.load(names[-1])):
                raise ValueError(f"{self.od_file} doesn't match {names[-1]}; rerun tools/import_stdn_od.py")
            aux = [[od_maps[rows] for _, rows in group] for group in parts]
        windows = lambda i: ConcatDataset([FlowWindows(a, t_in, t_out, lo, hi, e, x)
                                           for a, e, x in zip(frames[i], ext[i], aux[i])])
        extra_in, extra_lags, trend_lags = 0, (), frozenset()
        if period_shift and not periods:
            raise ValueError("--period-shift needs --periods")
        if periods:
            if external:
                raise ValueError("--periods together with --external isn't supported")
            steps, trend_lags = self.period_steps(periods, period_shift or 0)
            extra_in = len(steps) * t_out
            extra_lags = tuple(p for p in steps for _ in range(t_out))   # PeriodicFlowWindows' order

            def windows(i):   # the whole file, so the periodic frames can reach back past the split
                return ConcatDataset([PeriodicFlowWindows(files.load(n), t_in, t_out, lo, hi, steps,
                                                          rows.start or 0, rows.stop,
                                                          files.load(self.od_file) if od else None)
                                      for n, rows in parts[i]])
        split = dict(train_files=train_files, val_files=val_files, test_files=test_files,
                     train_size=train_size, val_size=val_size, test_size=test_size)
        if external:
            split["external"] = True
        if periods:
            split["periods"] = list(periods)
            if period_shift:
                split["period_shift"] = period_shift
        if test_days:
            split["test_days"] = test_days
        if od:
            split["od"] = True
        return Splits(name, limit(windows(0), train_size, "train"),
                      limit(windows(1), val_size, "val"), limit(windows(2), test_size, "test"),
                      lo, hi, self.channels, t_in, t_out, split, spatial_features=self._spatial(),
                      ext_dim=(len(ext_data.CALENDAR_NAMES) if self.calendar else ext_data.DIM) if external else 0,
                      extra_in=extra_in,
                      extra_lags=extra_lags, trend_lags=trend_lags, aux_in=2 if od else 0)

    def _spatial(self):
        if self.spatial_file is None:
            return None
        f = np.asarray(files.load(self.spatial_file), dtype=np.float32)
        f = f.reshape(f.shape[0], -1).T
        return (f - f.mean(0)) / (f.std(0) + 1e-6)

    def _by_time(self, names, test_days=None):
        *head, last = names
        test_frames = round(test_days * self.steps_per_day) if test_days else self.test_frames
        cut_test = len(files.load(last)) - test_frames
        cut_val = cut_test - self.val_frames
        if cut_val <= 0:
            raise ValueError(f"{last} is too short for the val and test periods")
        everything = slice(None)
        train = [(n, everything) for n in head] + [(last, slice(0, cut_val))]
        return train, [(last, slice(cut_val, cut_test))], [(last, slice(cut_test, None))]

    def _by_file(self, names, train_files, val_files, test_files):
        if not (train_files and val_files and test_files):
            raise ValueError("splitting by file needs train_files, val_files and test_files")
        groups = [self._match(names, specs) for specs in (train_files, val_files, test_files)]
        shared = (set(groups[0]) & set(groups[1])) | (set(groups[0]) & set(groups[2])) \
            | (set(groups[1]) & set(groups[2]))
        if shared:
            raise ValueError(f"file(s) in more than one split: {sorted(shared)}")
        return [[(n, slice(None)) for n in group] for group in groups]

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
               train_size=None, val_size=None, test_size=None, external=False):
        if external:
            raise ValueError("Moving MNIST has no external data; drop --external")
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


class _ArrayWindows(Dataset):
    """Fixed (x, y) windows, (S, L, C, H, W) raw counts, scaled to 0-1 by lo / hi on access."""

    def __init__(self, x, y, lo, hi):
        self.x, self.y, self.lo, self.span = x, y, lo, max(hi - lo, 1e-8)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        scale = lambda a: torch.from_numpy((np.asarray(a, dtype=np.float32) - self.lo) / self.span)
        return scale(self.x[i]), scale(self.y[i])


class StsslData:
    """The ST-SSL release (Ji et al., AAAI'23): the paper's own windows, imported with
    tools/import_stssl.py. Next step only; the input is the recent frames (t_in = the run of offsets
    ending at 0) plus, as the periodic frames, the target time +-k steps on each of the 3 days before."""
    SOURCE = "download https://github.com/Echo-Ji/ST-SSL_Dataset and run tools/import_stssl.py on it"

    def __init__(self, title, folder, channels=("Inflow", "Outflow")):
        self.title, self.folder, self.channels = title, folder, channels

    def _path(self, name):
        path = os.path.join(files.ROOT, "STSSL", self.folder, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"no {path}; {self.SOURCE}")
        return path

    @property
    def offsets(self):
        return np.load(self._path("offsets.npy")).tolist()

    @property
    def horizon(self):
        offs = self.offsets
        recent = 1
        while recent < len(offs) and offs[-recent - 1] == offs[-recent] - 1:
            recent += 1
        return recent, 1

    def splits(self, name, t_in, t_out, train_files=None, val_files=None, test_files=None,
               train_size=None, val_size=None, test_size=None, external=False, periods=None,
               period_shift=0, test_days=None):
        if external or periods or period_shift or test_days or train_files or val_files or test_files:
            raise ValueError(f"{self.title} is a fixed set of windows: no --external / --periods / "
                             "--period-shift / --test-days / --*-files")
        if (t_in, t_out) != self.horizon:
            raise ValueError(f"{self.title} windows are fixed at --t-in {self.horizon[0]} --t-out 1 "
                             f"(plus {len(self.offsets) - self.horizon[0]} periodic frames)")
        arrays = {f"{s}_{k}": np.load(self._path(f"{s}_{k}.npy"), mmap_mode="r")
                  for s in ("train", "val", "test") for k in ("x", "y")}
        lo = float(min(arrays["train_x"].min(), arrays["train_y"].min()))
        hi = float(max(arrays["train_x"].max(), arrays["train_y"].max()))
        part = lambda s: _ArrayWindows(arrays[f"{s}_x"], arrays[f"{s}_y"], lo, hi)
        offs = self.offsets
        extra_lags = tuple(1 - o for o in offs[:len(offs) - t_in])   # steps before the target
        split = dict(train_size=train_size, val_size=val_size, test_size=test_size)
        return Splits(name, limit(part("train"), train_size, "train"), limit(part("val"), val_size, "val"),
                      limit(part("test"), test_size, "test"), lo, hi, self.channels, t_in, t_out, split,
                      extra_in=len(extra_lags), extra_lags=extra_lags)


class SyntheticData:
    title = "Synthetic"
    channels = ("Inflow", "Outflow")
    horizon = (4, 4)

    def splits(self, name, t_in, t_out, train_files=None, val_files=None, test_files=None,
               train_size=None, val_size=None, test_size=None, external=False, frames=1500, size=32, seed=0):
        if external:
            raise ValueError("the synthetic data has no external data; drop --external")
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


BOUSAI_SOURCE = ("apply for Yahoo! Bousai Crowd Data at https://randd.yahoo.co.jp/en/softwaredata, then "
                 "convert flowio_*.npy with tools/import_grid.py (see its --help)")

DATASETS = {
    "taxibj": GridFlowData("TaxiBJ", r"TaxiBJ/TaxiBJ20\d\d(_\d+)?\.npy", ("Inflow", "Outflow"),
                           steps_per_day=48, test_days=28, val_days=28, has_external=True),
    "bikenyc": GridFlowData("BikeNYC", r"BikeNYC/flow_data\.npy", ("Inflow", "Outflow"),
                            steps_per_day=24, test_days=10, val_days=10,
                            spatial_file="BikeNYC/poi_data.npy"),
    # NYC taxi trips and Citi Bike, Jan-Mar 2015 / Jul-Aug 2016, from STDN (Yao et al., AAAI'19);
    # the last 20 days are test, as in STDN. STDN calls the channels start / end volume.
    "nyctaxi": GridFlowData("NYCTaxi", r"NYCTaxi/volume\.npy", ("Start", "End"),
                            steps_per_day=48, test_days=20, val_days=10, od_file="NYCTaxi/od_aux.npy",
                            calendar=("2015-01-01", ("2015-01-01", "2015-01-19", "2015-02-16"))),
    "nycbike2": GridFlowData("NYCBike2", r"NYCBike2/volume\.npy", ("Start", "End"),
                             steps_per_day=48, test_days=20, val_days=10, od_file="NYCBike2/od_aux.npy",
                             calendar=("2016-07-01", ("2016-07-04",))),
    # 89 days of Beijing taxi flow; the published file is already scaled to 0-1, so its "raw
    # units" are that scale, not counts.
    "taxibj21": GridFlowData("TaxiBJ21", r"TaxiBJ/TaxiBJ21\.npy", ("Inflow", "Outflow"),
                             steps_per_day=48, test_days=14, val_days=14),
    # Yahoo! Bousai Crowd Data (DeepCrowd): smartphone GPS crowd flow in Tokyo / Osaka, 450 m cells,
    # 2017-04-01 .. 07-09. Given out on application only; import with tools/import_grid.py.
    "bousai_tokyo": GridFlowData("BousaiTYO", r"Bousai/flowio_tokyo\.npy", ("Inflow", "Outflow"),
                                 steps_per_day=48, test_days=14, val_days=14,
                                 source=BOUSAI_SOURCE),
    "bousai_osaka": GridFlowData("BousaiOSA", r"Bousai/flowio_osaka\.npy", ("Inflow", "Outflow"),
                                 steps_per_day=48, test_days=14, val_days=14,
                                 source=BOUSAI_SOURCE),
    # ST-SSL (AAAI'23) windows: the paper's 7:1:2 split, input = recent frames + the target time
    # +-4 steps (+-2 for NYCBike1) on each of the 3 days before. NYCTaxi / NYCBike2 are STDN's data,
    # NYCBike1 / BJTaxi ST-ResNet's. Score as the paper does: MAE / MAPE on cells with true > 5.
    "stssl_nyctaxi": StsslData("ST-SSL NYCTaxi", "NYCTaxi", ("Start", "End")),
    "stssl_nycbike1": StsslData("ST-SSL NYCBike1", "NYCBike1"),
    "stssl_nycbike2": StsslData("ST-SSL NYCBike2", "NYCBike2", ("Start", "End")),
    "stssl_bjtaxi": StsslData("ST-SSL BJTaxi", "BJTaxi"),
    "mmnist": MovingMNISTData(),
    "synthetic": SyntheticData(),
}


def get_splits(dataset="taxibj", t_in=None, t_out=None, **split):
    spec = DATASETS[dataset]
    t_in = t_in or spec.horizon[0]
    t_out = t_out or spec.horizon[1]
    split = {k: v for k, v in split.items() if k in SPLIT_KEYS}
    if not split.get("periods"):
        split.pop("periods", None)
    if not split.get("period_shift"):
        split.pop("period_shift", None)
    if not split.get("test_days"):
        split.pop("test_days", None)
    if not split.get("od"):
        split.pop("od", None)
    elif not isinstance(spec, GridFlowData):
        raise ValueError(f"--test-days needs a grid flow dataset with a time axis, not {dataset}")
    elif not isinstance(spec, GridFlowData):
        raise ValueError(f"--periods needs a grid flow dataset with a time axis, not {dataset}")
    return spec.splits(dataset, t_in, t_out, **split)


def scale_of(ds):
    first = base_datasets(ds)[0]
    return first.lo, first.lo + first.span


def _size(text):
    return float(text) if any(c in text for c in ".eE") else int(text)


def add_data_args(ap):
    g = ap.add_argument_group(
        "data",
        "Default split: the last weeks or days of the data are test (4 weeks for taxibj, 10 days "
        "for bikenyc; see the README for the others), the same or a shorter period before that is "
        "val, the rest is train. Name files to split by file instead, "
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
    g.add_argument("--test-days", type=float, metavar="D",
                   help="test on the last D days instead of the dataset's default (val stays the same length "
                        "and moves back; e.g. 14 for DeepSTN+'s BikeNYC split)")
    g.add_argument("--periods", nargs="+", metavar="P",
                   help="also feed, for every target frame, the frame P earlier: day, week or a number of "
                        "steps (e.g. --periods day week). Windows without that much history are dropped")
    g.add_argument("--period-shift", type=int, default=0, metavar="Q",
                   help="also feed the frames Q steps either side of each period (STDN's shifted history), "
                        "e.g. --periods 1day 2day 3day week --period-shift 1: 12 extra frames per target")
    g.add_argument("--od", action="store_true",
                   help="nyctaxi / nycbike2: two more input maps per frame from STDN's origin-destination "
                        "flows (tools/import_stdn_od.py): trips under way by origin, and where they are expected "
                        "to arrive next step. Every model takes them; the targets stay Start / End")
    g.add_argument("--external", action="store_true",
                   help="stresnet only: feed TaxiBJ's external data (TaxiBJext: day of week, weekend, "
                        "time of day, temperature, wind, weather type) to its external component")
    return g


def splits_from_args(args):
    return get_splits(args.dataset, args.t_in, args.t_out,
                      **{k: getattr(args, k, None) for k in SPLIT_KEYS if getattr(args, k, None)})
