"""Score PEMS04 / PEMS08 checkpoints the way the PeMS papers do (STSGCN, STID, STAEformer, STD-MAE): MAE,
RMSE and MAPE of the flow over all 12 forecast steps together, and at steps 3 / 6 / 12 (15 / 30 / 60 min).
BjTT (road speed, 4-min steps, 15 out): the average over the 15 steps and steps 5 / 10 / 15 (20 / 40 / 60 min),
the BjTT paper's horizons, with MAE / RMSE at each.
The masked numbers leave out readings whose true value is 0 (the BasicTS / STAEformer masked metrics with
null value 0); the "all" MAE / RMSE keep them. Seeds of a run are averaged (mean and sample sd).

  python tools/eval_pems.py checkpoints/workspaces/pems_pems08 [--match REGEX]
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # Apple GPUs: run any op MPS lacks on the CPU
import argparse
import collections
import glob
import re
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_published import predictions                       # noqa: E402
from trafficpred.utils import pick_device                     # noqa: E402

# averaged over the 12 steps, masked: (MAE, RMSE, MAPE %), from the papers' own tables
PUBLISHED = {
    "pems04": [("STID (CIKM'22), in STAEformer's table", (18.38, 29.95, 12.04)),
               ("STAEformer (CIKM'23)", (18.22, 30.18, 11.98)),
               ("STD-MAE (IJCAI'24, long-history pre-training)", (17.80, 29.25, 11.97))],
    "pems08": [("STID (CIKM'22), in STAEformer's table", (14.21, 23.28, 9.27)),
               ("STAEformer (CIKM'23)", (13.46, 23.25, 8.88)),
               ("STD-MAE (IJCAI'24, long-history pre-training)", (13.44, 22.47, 8.76))],
}
# BjTT baselines: MAE / RMSE at steps 5 / 10 / 15 (20 / 40 / 60 min), the sota2.com BjTT leaderboards
# (traffic-prediction-on-bjtt-t-{5-20,10-40,15-60}-min-test, rows from CrossTrafficLLM, arXiv 2601.06042; its GNN rows
# match the BjTT paper's). Protocol not stated there; the BjTT paper is 70 / 10 / 20 on all three months. The BjTT
# paper's own text-conditioned LDM (arXiv 2403.05029, Table II) isn't on the leaderboard and is added last.
PUBLISHED_HORIZONS = {
    "bjtt": [
        ("CrossTrafficLLM, with text", ((3.25, 4.86), (3.37, 5.11), (3.55, 5.36))),
        ("CrossTrafficLLM, no text", ((3.31, 4.92), (3.45, 5.19), (3.64, 5.46))),
        ("Crossformer", ((3.39, 5.03), (3.52, 5.31), (3.79, 5.51))),
        ("TimeXer", ((3.43, 5.21), (3.61, 5.48), (3.74, 5.63))),
        ("Informer", ((3.55, 5.36), (3.84, 5.85), (4.02, 6.13))),
        ("STGCN", ((3.56, 5.45), (3.85, 5.93), (4.0, 6.19))),
        ("STG-NCDE", ((3.56, 5.53), (3.83, 6.09), (4.05, 6.52))),
        ("MTGNN", ((3.6, 5.6), (3.74, 5.91), (3.89, 6.19))),
        ("GWN", ((3.61, 5.6), (3.92, 6.19), (4.16, 6.61))),
        ("STSGCN", ((3.66, 5.74), (4.01, 6.22), (4.15, 6.63))),
        ("ASTGCN", ((3.79, 5.87), (4.15, 6.54), (4.44, 7.02))),
        ("iTransformer", ((3.83, 5.91), (3.84, 5.93), (3.86, 5.99))),
        ("LDM, text-conditioned diffusion (BjTT paper)", ((3.30, 5.08), (3.26, 5.07), (3.24, 5.00))),
    ],
}
HORIZONS = {
    "bjtt": [("STGCN (BjTT paper)", ((3.56, 5.45), (3.85, 5.93), (4.00, 6.19))),
             ("MTGNN (BjTT paper)", ((3.60, 5.60), (3.74, 5.91), (3.89, 6.19))),
             ("LDM, text-conditioned diffusion (BjTT paper)", ((3.30, 5.08), (3.26, 5.07), (3.24, 5.00))),
             ("CrossTrafficLLM (arXiv 2601.06042, own split)", ((3.25, 4.86), (3.37, 5.11), (3.55, 5.36)))],
}
HORIZONS = {"bjtt": (5, 10, 15)}   # forecast steps reported on their own (1-based); default 3 / 6 / 12


def horizons(dataset):
    return HORIZONS.get(dataset, (3, 6, 12))


def scores(pred, true, steps_out=(3, 6, 12)):
    pred, true = pred.flatten(2), true.flatten(2)                    # (N, T, sensors)
    out = {}
    for name, steps in [("avg", slice(None))] + [(f"h{k}", slice(k - 1, k)) for k in steps_out]:
        p, t = pred[:, steps], true[:, steps]
        keep = t > 0.5                                               # true 0 (counts); float error stays away
        e = (p - t)[keep]
        out[name] = (e.abs().mean().item(), e.pow(2).mean().sqrt().item(), (e.abs() / t[keep]).mean().item() * 100)
        if name == "avg":
            out["all"] = ((p - t).abs().mean().item(), (p - t).pow(2).mean().sqrt().item())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--match")
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    device = pick_device()
    paths = []
    for p in args.paths:
        paths += sorted(glob.glob(os.path.join(p, "*.pt"))) if os.path.isdir(p) else [p]
    paths = [p for p in paths if not p.endswith(".last.pt")]
    paths = [p for p in paths if os.path.exists(p[:-3] + ".done") or not glob.glob(os.path.join(os.path.dirname(p), "*.done"))]
    if args.match:
        paths = [p for p in paths if re.search(args.match, os.path.basename(p))]
    runs, dataset = collections.defaultdict(list), None
    for path in paths:
        ck, pred, true = predictions(path, device, args.split, args.batch_size)
        if dataset and ck["dataset"] != dataset:
            raise SystemExit("give runs of one dataset at a time")
        dataset = ck["dataset"]
        runs[re.sub(r"_seed\d+$", "", os.path.basename(path)[:-3])].append(scores(pred, true, horizons(dataset)))
        print(f"scored {os.path.basename(path)}", file=sys.stderr, flush=True)
    if not runs:
        raise SystemExit("no finished runs matched")

    def ms(vals):
        return f"{st.mean(vals):6.2f}" + (f" ±{st.stdev(vals):4.2f}" if len(vals) > 1 else "      ")

    hs = horizons(dataset)
    what = "speed" if dataset == "bjtt" else "flow"
    print(f"{dataset} {args.split}: {what}, {pred.shape[1]} steps, masked (true > 0) unless 'all'; mean ± sd over seeds")
    print(f"  {'run':52}{'n':>2}  {'MAE':>12}{'RMSE':>12}{'MAPE %':>12}   {'all MAE':>8}{'all RMSE':>9}   "
          + "".join(f"{f'MAE@{k}':>8}" for k in hs) + "".join(f"{f'RMSE@{k}':>8}" for k in hs))
    order = sorted(runs, key=lambda r: st.mean(s["avg"][0] for s in runs[r]))
    for r in order:
        s = runs[r]
        cols = [ms([x["avg"][k] for x in s]) for k in range(3)]
        print(f"  {r[:52]:52}{len(s):2d}  " + "".join(f"{c:>12}" for c in cols)
              + f"   {st.mean(x['all'][0] for x in s):8.2f}{st.mean(x['all'][1] for x in s):9.2f}   "
              + "".join(f"{st.mean(x[f'h{k}'][0] for x in s):8.2f}" for k in hs)
              + "".join(f"{st.mean(x[f'h{k}'][1] for x in s):8.2f}" for k in hs))
    if args.split == "test" and dataset in PUBLISHED:
        print("\npublished (12-step average, masked): MAE / RMSE / MAPE")
        for name, (mae, rmse, mape) in PUBLISHED[dataset]:
            print(f"  {name:52}{mae:8.2f}{rmse:8.2f}{mape:8.2f}")
    if args.split == "test" and dataset in PUBLISHED_HORIZONS:
        print(f"\npublished MAE / RMSE at steps {' / '.join(map(str, hs))} (the BjTT papers don't say masked or not)")
        for name, vals in PUBLISHED_HORIZONS[dataset]:
            print(f"  {name:52}" + "   ".join(f"{m:5.2f} / {r:5.2f}" for m, r in vals))


if __name__ == "__main__":
    main()
