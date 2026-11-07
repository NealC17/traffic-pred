"""Data utilities for STDCformer.

Expected dataset format (.npz):
  flow          float (T, N, f)   inflow/outflow per region per step
  timestamps    int64 (T,)        unix seconds of each step start
  adj           float (N, N)      region adjacency (taxi-zone contiguity, distance, ...)
  holiday       int   (T,)        optional, 0/1
  temporal_feat float (T, t)      optional, temperature / weather one-hot ...
  spatial_feat  float (N, s)      optional, POI counts / house prices / crime counts ...
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Laplacian positional encoding (spatial confounder, Sec. 3.2)
# ---------------------------------------------------------------------------
def laplacian_pe(adj: np.ndarray, k: int) -> np.ndarray:
    """k smallest non-trivial eigenvectors of the symmetric normalized Laplacian."""
    A = np.asarray(adj, dtype=np.float64)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    deg = A.sum(1)
    d_inv_sqrt = np.where(deg > 0, deg ** -0.5, 0.0)
    L = np.eye(len(A)) - d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    evals, evecs = np.linalg.eigh(L)
    pe = evecs[:, 1:k + 1]
    if pe.shape[1] < k:  # tiny graphs
        pe = np.pad(pe, ((0, 0), (0, k - pe.shape[1])))
    return pe.astype(np.float32)


class StandardScaler:
    def __init__(self, mean, std):
        self.mean, self.std = float(mean), float(std) if std > 0 else 1.0

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


# ---------------------------------------------------------------------------
# Windowed dataset
# ---------------------------------------------------------------------------
class CrowdFlowWindows(Dataset):
    def __init__(self, flow_norm, flow_raw, tod, dow, hol, exo, starts, T_p, T_f):
        self.flow_norm, self.flow_raw = flow_norm, flow_raw
        self.tod, self.dow, self.hol, self.exo = tod, dow, hol, exo
        self.starts, self.T_p, self.T_f = starts, T_p, T_f

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        xs, ys = slice(s, s + self.T_p), slice(s + self.T_p, s + self.T_p + self.T_f)
        item = {
            "x": self.flow_norm[xs],
            "y": self.flow_raw[ys],
            "tod_x": self.tod[xs], "tod_y": self.tod[ys],
            "dow_x": self.dow[xs], "dow_y": self.dow[ys],
            "hol_x": self.hol[xs], "hol_y": self.hol[ys],
        }
        if self.exo is not None:
            item["exo_x"], item["exo_y"] = self.exo[xs], self.exo[ys]
        return item


def build_dataloaders(npz_path: str, T_p=6, T_f=6, lap_dim=8, batch_size=64,
                      split=(0.7, 0.1, 0.2), steps_per_day=24, num_workers=0):
    """Chronological 7:1:2 split (paper). Scaler fit on the training span only."""
    d = np.load(npz_path)
    flow = d["flow"].astype(np.float32)
    T, N, f = flow.shape
    ts = d["timestamps"].astype(np.int64)
    step_sec = 86400 // steps_per_day
    tod = ((ts % 86400) // step_sec).astype(np.int64)
    dow = (((ts // 86400) + 3) % 7).astype(np.int64)   # 1970-01-01 was Thursday -> Monday=0
    hol = d["holiday"].astype(np.int64) if "holiday" in d else np.zeros(T, np.int64)
    exo = d["temporal_feat"].astype(np.float32) if "temporal_feat" in d else None
    sfeat = d["spatial_feat"].astype(np.float32) if "spatial_feat" in d else None

    n_win = T - T_p - T_f + 1
    n_tr, n_va = int(n_win * split[0]), int(n_win * split[1])
    starts = np.arange(n_win)
    tr, va, te = starts[:n_tr], starts[n_tr:n_tr + n_va], starts[n_tr + n_va:]

    train_span = flow[: n_tr - 1 + T_p + T_f]   # every step seen by a training window
    scaler = StandardScaler(train_span.mean(), train_span.std())
    flow_norm = scaler.transform(flow).astype(np.float32)

    if exo is not None:  # standardize continuous exogenous on train span
        mu, sd = exo[:n_tr + T_p].mean(0), exo[:n_tr + T_p].std(0) + 1e-6
        exo = (exo - mu) / sd
    if sfeat is not None:
        sfeat = (sfeat - sfeat.mean(0)) / (sfeat.std(0) + 1e-6)

    t = lambda a: torch.from_numpy(a) if a is not None else None
    tensors = [t(flow_norm), t(flow), t(tod), t(dow), t(hol), t(exo)]
    mk = lambda idx, shuffle: DataLoader(CrowdFlowWindows(*tensors, idx, T_p, T_f),
                                         batch_size=batch_size, shuffle=shuffle,
                                         num_workers=num_workers, drop_last=False)
    static = {
        "lap_pe": torch.from_numpy(laplacian_pe(d["adj"], lap_dim)),
        "spatial_feat": t(sfeat),
    }
    meta = dict(num_nodes=N, in_dim=f, temporal_feat_dim=0 if exo is None else exo.shape[1],
                spatial_feat_dim=0 if sfeat is None else sfeat.shape[1])
    # paper: "the training and validation sets are shuffled"
    return mk(tr, True), mk(va, True), mk(te, False), scaler, static, meta


# ---------------------------------------------------------------------------
# Metrics (computed on de-normalized values)
# ---------------------------------------------------------------------------
def masked_metrics(pred: torch.Tensor, true: torch.Tensor, mape_thresh: float = 1.0):
    """MAE / RMSE over all entries; MAPE over entries with true > mape_thresh
    (crowd-flow series contain many zeros, which make raw MAPE undefined)."""
    err = pred - true
    mae = err.abs().mean().item()
    rmse = err.pow(2).mean().sqrt().item()
    m = true.abs() > mape_thresh
    mape = (err[m].abs() / true[m].abs()).mean().item() if m.any() else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# ---------------------------------------------------------------------------
# Synthetic data with explicit spatial & temporal confounders
# ---------------------------------------------------------------------------
def make_synthetic(path: str, num_nodes=40, days=120, steps_per_day=24, seed=0):
    """Generates hourly in/out flows driven by:
       - spatial confounders: region 'type' (residential vs commercial) and POI density
       - temporal confounders: time of day, weekday/weekend, holidays, rain
       - graph diffusion between neighbouring regions
    so that the confounders genuinely affect both past and future flows."""
    rng = np.random.default_rng(seed)
    N, T = num_nodes, days * steps_per_day
    pos = rng.random((N, 2))
    dist = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    adj = ((dist < 0.25) & (dist > 0)).astype(np.float32)
    for i in range(N):  # guarantee connectivity to nearest neighbour
        j = np.argsort(dist[i])[1]
        adj[i, j] = adj[j, i] = 1.0

    commercial = rng.random(N)                       # 0 = residential, 1 = commercial
    poi = rng.gamma(2.0, 1.0, N)
    spatial_feat = np.stack([commercial, poi, rng.random(N)], 1).astype(np.float32)

    start = 1698796800                               # 2023-11-01 00:00 UTC
    ts = start + np.arange(T) * (86400 // steps_per_day)
    hour = (ts % 86400) // 3600
    dow = ((ts // 86400) + 3) % 7
    weekend = (dow >= 5).astype(np.float32)
    holiday = np.zeros(T, np.int64)
    for dday in rng.choice(days, size=max(1, days // 30), replace=False):
        holiday[dday * steps_per_day:(dday + 1) * steps_per_day] = 1
    rain = (np.convolve(rng.random(T) < 0.03, np.ones(6), "same") > 0).astype(np.float32)
    temp = 10 + 8 * np.sin(2 * np.pi * (np.arange(T) / steps_per_day - 15) / 365) \
           + 5 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 1, T)
    temporal_feat = np.stack([temp, rain], 1).astype(np.float32)

    gauss = lambda h, mu, s: np.exp(-0.5 * ((h - mu) / s) ** 2)
    morning, evening = gauss(hour, 8, 1.5), gauss(hour, 18, 2.0)
    offday = np.clip(weekend + holiday, 0, 1)

    base = 20 * poi[None]
    # commercial: inflow in the morning, outflow in the evening; residential reversed
    inflow = base * (0.3 + (commercial[None] * morning[:, None] + (1 - commercial[None]) * evening[:, None])
                     * (1 - 0.6 * offday[:, None]) + 0.4 * offday[:, None] * gauss(hour, 14, 3)[:, None])
    outflow = base * (0.3 + (commercial[None] * evening[:, None] + (1 - commercial[None]) * morning[:, None])
                      * (1 - 0.6 * offday[:, None]) + 0.4 * offday[:, None] * gauss(hour, 16, 3)[:, None])
    flow = np.stack([inflow, outflow], -1) * (1 - 0.3 * rain[:, None, None])

    # graph diffusion + AR(1) noise
    P = adj / adj.sum(1, keepdims=True)
    noise = np.zeros((N, 2))
    for tt in range(T):
        noise = 0.8 * noise + rng.normal(0, 2.0, (N, 2))
        flow[tt] = 0.8 * flow[tt] + 0.2 * P @ flow[tt] + noise
    flow = np.clip(flow, 0, None).astype(np.float32)

    np.savez(path, flow=flow, timestamps=ts.astype(np.int64), adj=adj, holiday=holiday,
             temporal_feat=temporal_feat, spatial_feat=spatial_feat)
    return path
