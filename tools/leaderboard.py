import argparse
import csv
import os
import secrets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACES = os.path.join(ROOT, "checkpoints", "workspaces")
COLUMNS = ["run", "test RMSE", "test MAE", "test MSE"]
ID_FILE = "leaderboard.wandb_id"


def latest_workspace():
    with open(os.path.join(WORKSPACES, "LATEST")) as f:
        return os.path.join(WORKSPACES, f.readline().strip())


def leaderboard_rows(results_path):
    with open(results_path) as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r.get("test_rmse")]
    baseline = [r for r in rows if r["run"].startswith("copy_last")]
    models = sorted((r for r in rows if not r["run"].startswith("copy_last")), key=lambda r: float(r["test_rmse"]))
    return [[r["run"], round(float(r["test_rmse"]), 2), round(float(r["test_mae"]), 2), round(float(r["test_mse"]), 1)]
            for r in baseline + models]


def print_rows(rows):
    print("| " + " | ".join(COLUMNS) + " |")
    print("|---|---|---|---|")
    for row in rows:
        print(f"| {row[0]} | {row[1]:.2f} | {row[2]:.2f} | {row[3]:.1f} |")


def start_run(wandb, workspace, name):
    id_path = os.path.join(workspace, ID_FILE)
    run_id = open(id_path).read().strip() if os.path.exists(id_path) else None
    kwargs = dict(project=os.environ.get("WANDB_PROJECT", "traffic-pred"), name="leaderboard", group=name,
                  job_type="leaderboard", dir=ROOT)
    try:
        run = wandb.init(id=run_id or secrets.token_hex(4), resume="allow", **kwargs)
    except Exception as e:
        if run_id is None:
            raise
        print(f"[leaderboard] couldn't resume W&B run {run_id} ({e}); starting a new one")
        run = wandb.init(id=secrets.token_hex(4), **kwargs)
    with open(id_path, "w") as f:
        f.write(run.id + "\n")
    return run


def main():
    ap = argparse.ArgumentParser(description="Post a workspace's test RMSE / MAE / MSE table (from results.tsv) "
                                             "to W&B as the 'leaderboard' table of the workspace's group.")
    ap.add_argument("--workspace", help="workspace directory or name (default: the most recently used)")
    ap.add_argument("--print-only", action="store_true", help="print the table, don't touch W&B")
    args = ap.parse_args()
    workspace = args.workspace or latest_workspace()
    if not os.path.isdir(workspace):
        workspace = os.path.join(WORKSPACES, workspace)
    name = os.path.basename(os.path.normpath(workspace))
    results = os.path.join(workspace, "results.tsv")
    if not os.path.exists(results):
        raise SystemExit(f"no results.tsv in {workspace}")
    rows = leaderboard_rows(results)
    print_rows(rows)
    if args.print_only or os.environ.get("WANDB_MODE") == "disabled":
        return
    import wandb
    run = start_run(wandb, workspace, name)
    run.log({"leaderboard": wandb.Table(columns=COLUMNS, data=rows)})
    run.summary["runs"] = len(rows)
    print(f"[leaderboard] {len(rows)} runs -> {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
