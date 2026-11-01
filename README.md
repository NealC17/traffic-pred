# traffic-pred

Citywide crowd-flow prediction on TaxiBJ (inflow / outflow on a 32×32 grid of Beijing):
given the last 4 half-hour frames, predict the next 4. Compares DistVolNet (predicts
*where* traffic goes and *how much* separately) against a CNN, an MLP, PredFormer (ours
and the official code) and a copy-last-frame reference, each trained with MSE and, where
the model's output is non-negative, with a Sinkhorn-based loss.

## Setup

```bash
source setup.sh        # creates .env/, installs requirements.txt, clones the official PredFormer
wandb login            # once per machine (API key from wandb.ai/authorize)
```

Works on Linux (CUDA), macOS (Apple MPS) and CPU; `devices.py` picks the device.
The datasets are committed compressed (`datasets/**/*.npy.xz`); `data.py` unpacks them on
first import.

## Train everything

```bash
./train_all.sh --quick           # 1 epoch per run: checks the pipeline
./train_all.sh                   # full training (see --help for options)
./train_all.sh cnn distvol       # only some runs
```

| run | model | loss |
|---|---|---|
| `copy_last` | last frame repeated (reference, no training) | – |
| `cnn_mse`, `cnn_sinkhorn` | BaselineCNN | MSE / DistVolLoss (L1 + Sinkhorn + volume) |
| `distvol_mse`, `distvol_sinkhorn` | DistVolNet | MSE / DistVolLoss |
| `mlp_mse` | MLP | MSE |
| `predformer_mse` | PredFormer (`predformer.py`, TSST) | MSE |
| `predformer_official_mse` | PredFormer (official repo, STS) | MSE |

Outputs: `checkpoints/<run>.pt` (best weights), `checkpoints/logs/<run>.log`,
`checkpoints/results.tsv` (final test MSE, Sinkhorn and volume error of every model).
A run is skipped once finished (`checkpoints/<run>.done`), so a job that hits its time
limit can simply be resubmitted.

On ASU Sol:

```bash
sbatch --gres=gpu:1 --time=08:00:00 --mem=32G --cpus-per-task=4 --wrap="bash train_all.sh"
tail -f checkpoints/logs/<run>.log
```

## Metrics and Weights & Biases

Every model is scored by the same code (`tracking.py`), whatever its training loss:
MSE, Sinkhorn divergence (errors in *where* traffic goes), volume error (errors in *how
much*), MAE / RMSE in taxi counts, and all of them per prediction step (t+1 … t+4).

W&B project `traffic-pred` (override with `WANDB_PROJECT` / `WANDB_ENTITY`):
live training progress (`progress/percent`, ETA, train loss), test metrics every epoch,
prediction heatmaps, and per run the final `mse`, `sinkhorn`, `volume_err` plus their
ratio to copy-last (`vs_copy_last/*`, below 1 = better than copying). No network on the
node: `WANDB_MODE=offline`, then `wandb sync wandb/offline-run-*`. Turn off: `--no-wandb`.

## Look at predictions

```bash
python run_viz.py --ckpt checkpoints/predformer_mse.pt --browse
```

or in a notebook: `from viz_flow import browse_notebook`.
