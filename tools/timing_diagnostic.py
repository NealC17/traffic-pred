"""Are multi-step forecast errors timing errors (a peak a step early / late) or amplitude errors?

A forecast that runs tau steps late has error e(k) = P(k) - Y(k) ~ -tau * dY/dt(k). For every
(window, channel, cell) trajectory over the interior horizon steps, fit the one tau that best explains
its errors (least squares, no intercept) and see how much of the error it removes. Controls, each
fitted the same way with one parameter per trajectory:

  spatial   e ~ -u * dY/dx and e ~ -v * dY/dy (the better of the two; the prediction one cell off)
  null      e ~ -tau * (dY/dt of another window, same cell and step): what one free parameter per
            trajectory removes by chance
  scale     e ~ a * Y (the forecast a constant factor too high / low over the horizon: amplitude)
  offset    e ~ a (a constant bias over the horizon: amplitude)

The spatial gradient at a cell rises and falls with the city's daily cycle, as Y does, so compare
"spatial" with "scale" before reading it as displacement.

Timing errors are real if "timing" clearly beats both. Also printed: the systematic lag per step
(one tau for all trajectories: does the forecast lag the truth more the further out it goes?).
Scored on trajectories whose mean true flow is >= --threshold, RMSE in raw units.

  python tools/timing_diagnostic.py checkpoints/workspaces/nyctaxi_out8/distvol_mse_nyctaxi_in4_out8.pt ...
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.eval_published import predictions


def fit_removed(e, g):
    """Per trajectory (last dim = steps) least-squares e ~ a * g; returns the residual."""
    a = (e * g).sum(-1, keepdim=True) / (g * g).sum(-1, keepdim=True).clamp_min(1e-9)
    return e - a * g


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--threshold", type=float, default=10)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    args = ap.parse_args()
    torch.manual_seed(0)
    for path in args.paths:
        _, pred, true = predictions(path, "cpu", args.split)            # (N, K, C, H, W)
        N, K = true.shape[:2]
        if K < 4:
            raise SystemExit(f"{path}: needs >= 4 forecast steps, has {K}")
        e = (pred - true)[:, 1:-1]                                        # interior steps 2 .. K-1
        dt = (true[:, 2:] - true[:, :-2]) / 2
        y = true[:, 1:-1]
        dx = torch.gradient(y, dim=-1)[0]
        dy = torch.gradient(y, dim=-2)[0]
        dt_null = dt[torch.randperm(N)]
        # trajectories: (N, C, H, W, steps)
        to_traj = lambda t: t.permute(0, 2, 3, 4, 1)
        e, dt, dx, dy, dt_null, y = map(to_traj, (e, dt, dx, dy, dt_null, y))
        keep = y.mean(-1) >= args.threshold                               # (N, C, H, W)
        rmse = lambda r: r[keep].pow(2).mean().sqrt().item()
        base = rmse(e)
        timing = rmse(fit_removed(e, dt))
        spatial = min(rmse(fit_removed(e, dx)), rmse(fit_removed(e, dy)))
        null = rmse(fit_removed(e, dt_null))
        scale = rmse(fit_removed(e, y))
        offset = rmse(fit_removed(e, torch.ones_like(y)))
        both = rmse(fit_removed(fit_removed(e, y), dx))          # spatial on top of the scale fit
        gain = lambda r: f"{r:7.3f} ({100 * (1 - r ** 2 / base ** 2):4.1f}% of MSE removed)"
        print(f"\n{os.path.basename(path)}  ({args.split}, steps 2..{K - 1}, {int(keep.sum())} trajectories)")
        print(f"  forecast RMSE          {base:7.3f}")
        print(f"  - per-trajectory timing  {gain(timing)}")
        print(f"  - spatial shift control  {gain(spatial)}")
        print(f"  - null control           {gain(null)}")
        print(f"  - amplitude: scale       {gain(scale)}")
        print(f"  - amplitude: offset      {gain(offset)}")
        print(f"  - scale, then spatial x  {gain(both)}")
        # systematic lag per step: one tau over all scored trajectories
        m = keep[..., None].expand_as(e)
        taus = [-(e[..., k][m[..., k]] * dt[..., k][m[..., k]]).sum() / dt[..., k][m[..., k]].pow(2).sum()
                for k in range(e.shape[-1])]
        print("  systematic lag (steps, + = forecast late) at t+" + " t+".join(
            f"{k + 2}: {t.item():+.2f}" for k, t in enumerate(taus)))


if __name__ == "__main__":
    main()
