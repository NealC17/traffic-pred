"""Saved W&B workspace views that always show the 10 best runs: one view per test metric (RMSE, MAE,
MSE), its runs sorted by that metric's current value, every chart limited to the first 10.

W&B redoes the sorting whenever runs log, so the charts follow the best 10 at any time, including
runs still training (their test value is the latest one logged). Needs  pip install wandb-workspaces.

  python tools/wandb_views.py                              the projects in PROJECTS
  python tools/wandb_views.py --project traffic-pred-nyctaxi --top 15
  python tools/wandb_views.py --project traffic-pred-nyctaxi --once     only if not made before

The view links are kept in checkpoints/workspaces/wandb_views.json; rerunning updates those views
instead of making new ones.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.path.join(ROOT, "checkpoints", "workspaces", "wandb_views.json")
PROJECTS = ["traffic-pred", "traffic-pred-nyctaxi", "traffic-pred-nycbike2"]
METRICS = ["RMSE", "MAE", "MSE"]


def build(ws, wr, entity, project, metric, top, url=None):
    key = f"{metric}/test"
    panels = [wr.LinePlot(title=f"{top} best: test {metric}", x="epoch", y=[key], max_runs_to_show=top,
                          layout=wr.Layout(w=12, h=8)),
              wr.BarPlot(title=f"{top} best: test {metric} (current)", metrics=[wr.SummaryMetric(key)],
                         orientation="h", max_runs_to_show=top, layout=wr.Layout(w=12, h=8))]
    for other in METRICS:
        if other != metric:
            panels.append(wr.LinePlot(title=f"test {other} of the {top} best by {metric}", x="epoch",
                                      y=[f"{other}/test"], max_runs_to_show=top, layout=wr.Layout(w=12, h=8)))
    view = ws.Workspace.from_url(url) if url else ws.Workspace(entity=entity, project=project)
    view.name = f"{top} best by test {metric}"
    view.sections = [ws.Section(name=f"{top} best by test {metric}", panels=panels, is_open=True)]
    view.runset_settings = ws.RunsetSettings(order=[ws.Ordering(ws.Summary(key), ascending=True)])
    return view


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", action="append", help=f"W&B project (repeatable; default {' '.join(PROJECTS)})")
    ap.add_argument("--entity", help="W&B entity (default: your default entity)")
    ap.add_argument("--top", type=int, default=10, help="runs per chart (default 10)")
    ap.add_argument("--once", action="store_true", help="skip projects that already have the views")
    args = ap.parse_args()
    try:
        import wandb
        import wandb_workspaces.reports.v2 as wr
        import wandb_workspaces.workspaces as ws
    except ImportError:
        raise SystemExit("needs wandb-workspaces:  pip install wandb-workspaces")
    entity = args.entity or wandb.Api().default_entity
    store = json.load(open(STORE)) if os.path.exists(STORE) else {}
    for project in args.project or PROJECTS:
        views = store.setdefault(f"{entity}/{project}", {})
        if args.once and all(m in views for m in METRICS):
            continue
        for metric in METRICS:
            try:
                view = build(ws, wr, entity, project, metric, args.top, views.get(metric))
                view.save()
            except Exception as e:   # e.g. a stored view that was deleted on W&B: make a new one
                if metric not in views:
                    print(f"[views] {project}: couldn't save the {metric} view ({e})", file=sys.stderr)
                    continue
                view = build(ws, wr, entity, project, metric, args.top)
                view.save()
            views[metric] = view.url
            print(f"[views] {project}: {view.name} -> {view.url}")
        os.makedirs(os.path.dirname(STORE), exist_ok=True)
        with open(STORE, "w") as f:
            json.dump(store, f, indent=1)


if __name__ == "__main__":
    main()
