# traffic-pred

Citywide crowd-flow prediction: given the last few frames of an inflow / outflow grid, predict
the next few. The main model, **DistVolNet**, predicts *where* traffic goes (a spatial
distribution) separately from *how much* there is (total volume), trained with L1 + a
Sinkhorn optimal-transport term + a volume term. It is compared against a CNN, ST-ResNet, an MLP,
PredFormer (ours and the official code), STDCformer, ASTGNN and a copy-last-frame reference.
Every model is trained and scored by the same code on the same windows.

## Setup

```bash
source setup.sh     # creates .env/, installs requirements.txt, clones PredFormer, ASTGNN and STDCformer into external/
wandb login         # once per machine
```

Works on CUDA, Apple MPS and CPU. Datasets are committed compressed (`datasets/**/*.npy.xz`) and
unpacked on first use.

## Layout

```
train.py                  train / score any model on any dataset
astgnn.py                 prepare data for, and score, the official ASTGNN code
visualize.py              plot or browse a checkpoint's test predictions
train_all.sh              every model / loss combination as one resumable session
setup.sh, requirements.txt

trafficpred/
  data/                   dataset registry (taxibj, bikenyc, mmnist, synthetic), windows, splits
  models/                 copy_last, cnn, distvol, distvolformer, stresnet, mlp, predformer, and wrappers for the
                          official predformer and stdcformer code (+ the registry in __init__.py)
  losses.py               mse / mae / sinkhorn (DistVolLoss)
  sinkhorn.py             debiased Sinkhorn divergence on a grid, split_flow
  metrics.py              evaluate(): the metrics every model is scored with
  training.py             the shared training loop (Trainer)
  tracking.py             Weights & Biases logging
  checkpoints.py          load_model(path) / splits_for(checkpoint)
  astgnn.py               ASTGNN data conversion helpers
  viz.py                  heatmaps, interactive browser, notebook browser

tools/                    compress_data.py, param_sizes.py, convert_checkpoints.py
notebooks/                visualize.ipynb
external/                 (gitignored) PredFormer, ASTGNN and STDCformer clones, ASTGNN experiments
datasets/                 compressed data
```

## Datasets

| `--dataset` | grid | step | default horizon | default split |
|---|---|---|---|---|
| `taxibj` | 2 × 32 × 32 | 30 min | 4 → 4 | test = last 4 weeks, val = 4 weeks before, train = rest |
| `bikenyc` | 2 × 21 × 12 | 1 h | 4 → 4 | test = last 10 days, val = 10 days before, train = rest |
| `mmnist` | 1 × 64 × 64 | – | 10 → 10 | train generated on the fly, val a fixed generated set, test `mnist_test_seq.npy` |
| `synthetic` | 2 × 32 × 32 | – | 4 → 4 | drifting Gaussian hotspots, 70 / 15 / 15; a smoke test that needs no files |

Every dataset accepts `--t-in / --t-out` and `--train-size / --val-size / --test-size` (keep the
first N windows, or a fraction like `0.5`). File-based datasets can also be split by file:
`--train-files 2013 --val-files 2014 --test-files 2015 2016`. The min-max scale always comes from
the training data only. Windows never cross file or split boundaries.

Moving MNIST downloads on first use. On a cluster node without internet run
`python -c "from trafficpred.data import mmnist; mmnist.download()"` on a login node first.

## Train

```bash
python train.py --model distvol --loss sinkhorn
python train.py --model distvolformer --loss sinkhorn
python train.py --model stdcformer --dataset bikenyc --amp
python train.py --model predformer --dataset mmnist --amp
python train.py --model cnn --dataset synthetic --epochs 1        # smoke test
python train.py --model stresnet --resume                          # continue <save>.last.pt
python train.py --help                                             # every option
```

| `--model` | losses (first = default) | defaults |
|---|---|---|
| `copy_last` | – | no training |
| `cnn`, `distvol` | sinkhorn, mse, mae | 50 epochs, AdamW 1e-3, wd 1e-4, clip 1, one-cycle |
| `distvolformer` | sinkhorn, mse, mae | same, but wd 1e-2 |
| `stresnet` | mse, sinkhorn, mae | same; closeness branch only, output starts at the mean training value |
| `mlp` | mse, mae | 50 epochs, wd 1e-2 |
| `predformer`, `predformer_official` | mse, mae | 200 epochs (2000 on mmnist), wd 1e-2, paper configs |
| `stdcformer` | masked_mae, mae, mse | official config: 120 epochs, batch 64, Adam, wd 1e-4, clip 5, plateau, early stop 50 |

Any training flag (`--epochs`, `--lr`, `--batch-size`, `--weight-decay`, `--clip`, `--schedule`,
`--patience`) overrides the model's default. The sinkhorn loss needs a non-negative output with a
distribution / volume split, so only `cnn`, `distvol`, `distvolformer` and `stresnet` take it.

Protocol: every epoch is scored on val, the checkpoint with the lowest val MSE (`--select`) is
kept, and at the end that checkpoint is scored on train, val and test (MSE, RMSE and MAE in raw
units, plus every metric below). The seed defaults to 42. Checkpoints go to
`checkpoints/<dataset>/<run>.pt` and hold the weights, everything needed to rebuild the model and
its data split, and the val / test metrics. `<run>.last.pt` is written every epoch for `--resume`,
which refuses to continue a run made with different settings.

### External data (`--external`, ST-ResNet on TaxiBJ only)

`--external` gives `stresnet` the paper's external component, fed from TaxiBJ's external files
(`TaxiBJext2013-2016.npy`, one row per half hour; `TaxiBJ2016_1` and `TaxiBJ2016_2` are the first
and second halves of `TaxiBJext2016`). Run names get `_ext`; `./train_all.sh --external` runs the
two ST-ResNet runs only. Other models refuse the flag.

Each target step gets 29 features:

- calendar (10): day of week (7, one-hot), weekend flag, time of day as sin / cos of the
  half-hour slot. Known in advance, so they are the target step's real values.
- weather (19): temperature and wind speed (standardised with the train split's mean and std) and
  the weather type (17, one-hot). Future weather isn't known at forecast time, so every target
  step uses the last input step's weather.

The features go through FC(10) -> ReLU -> FC(C*H*W) -> ReLU and are added to the fused branches
before the output sigmoid, as in the paper.

### DistVolFormer

`distvolformer` keeps DistVolNet's split into a spatial distribution and a total volume per
channel and step, but swaps the two encoders. The distribution branch is a transformer: the input
grid is cut into 2x2 patches (`--patch-size`), each patch is a token with a learned position
embedding, and 6 pre-norm blocks (`--num-blocks`, `--dim` 192, `--heads` 6, `--mlp-dim` 4*dim,
`--drop` 0.1) predict per-cell logits that a spatial softmax turns into the distribution. The
volume branch is a UNet (`--width` 64) over the same inputs plus the log input volumes; its pooled
code predicts the change in volume from the last input frame, and also conditions every
transformer block through adaptive layer norm. About 7.5M parameters on TaxiBJ (DistVolNet: 4.0M).

Grids that don't fit a model are adapted: PredFormer lowers its patch size to divide the grid
(3 on BikeNYC), DistVolNet pads its encoder input to a multiple of 4 and crops before the
spatial softmax, and DistVolFormer also pads to a multiple of its patch size.

### STDCformer on a grid

`stdcformer` runs the official model code (He et al., Information Fusion 2026,
[GeoX-Lab/STDCformer](https://github.com/GeoX-Lab/STDCformer), cloned unmodified into
`external/STDCformer`) inside this project's training loop, data and metrics, like
`predformer_official`. Their own script needs their NYC data files and never saves predictions,
so it can't be scored with `metrics.py`. Defaults follow their `STDC.yaml`: D 64, 5 encoder and 5
decoder blocks, 8 heads, Adam 1e-3 with weight decay 1e-4, batch 64, 120 epochs, gradient clip 5,
reduce-on-plateau and early stopping after 50 epochs, standard-scaled input, and their MAE that
ignores cells whose true value is 0 (`--loss masked_mae`).

On a grid every cell is a region linked to its 8 neighbours (`--neighbors 4` for 4), and the
Laplacian eigenvectors of that graph are part of the spatial confounder. BikeNYC adds its 9 POI
features; TaxiBJ has none. The windows carry no timestamps, so the temporal confounder is a one-hot
of each step's position in the window, which means the back-door gate can't tell rush hour from
night. The official code needs `t_in == t_out` and builds every cell-to-cell attention matrix
explicitly (1024 x 1024 per head and step on TaxiBJ), so the wrapper uses gradient checkpointing:
each encoder / decoder block is recomputed in the backward pass instead of keeping its activations
(BatchNorm running statistics are frozen during the recompute, so results are bit-for-bit the same).
That cuts memory about 4x, at roughly the same speed per sample. Measured peaks: BikeNYC batch 64
(the official size) 2.7 GB; TaxiBJ about 0.55 GB per sample, so batch 8 on an 8 GB GPU and the
official 64 needs about 35 GB (an A100). Lower `--batch-size` if it runs out; `--no-checkpointing`
turns it off.

## Train everything

```bash
./train_all.sh --workspace                         # new workspace (named by date and time), every run
./train_all.sh --workspace=baselines               # new workspace with a name
./train_all.sh                                     # continue the most recently used workspace
./train_all.sh --dataset bikenyc stdcformer astgnn # taxibj (default) or bikenyc; some runs only
./train_all.sh --quick                             # 1 epoch each: pipeline check
./train_all.sh --no_save --quick                   # same check, but saves no checkpoints
./train_all.sh --batch-size stdcformer=8           # one model's batch size (repeatable; N alone = all)
./train_all.sh --workspaces                        # list workspaces
./train_all.sh --help
```

A workspace lives in `checkpoints/workspaces/<name>/` and is one W&B group. Without `--workspace`
the most recently used workspace is continued: finished runs are skipped and interrupted ones
resume, so a Slurm job can just be resubmitted. Every model, ASTGNN included, uses seed 42 unless
`--seed` is given. Each run picks its checkpoint on val MSE, and that checkpoint is scored on the
train, val and test sets; `results.tsv` holds MSE, RMSE and MAE in raw units (taxis / bikes per
cell) for all three, plus the best epoch. With `--no_save` (`train.py --no-save`) every selected
run trains and is scored in memory only: no checkpoints, `.done` markers or `results.tsv` rows,
finished runs aren't skipped, and logs go to `logs/<run>.check.log`.

```bash
sbatch --gres=gpu:1 --time=08:00:00 --mem=32G --cpus-per-task=4 --wrap="bash train_all.sh"
```

## ASTGNN

The official code (Guo et al., TKDE 2021) runs unmodified from `external/ASTGNN`. Each grid cell is
a node; ASTGNN predicts one feature, so each channel is its own run. `astgnn.py` writes ASTGNN's
data file straight from this project's windows (same inputs, targets and order as every other
model) and needs `t_in == t_out`.

`python astgnn.py run --dataset bikenyc` does all three steps below in one command; it is also the
`astgnn` run in `train_all.sh`, where `--epochs N` sets ASTGNN's training epochs to N and its
fine-tuning epochs to N/2. Training goes through `trafficpred/astgnn_runner.py`, which sets the seed
(default 42), runs ASTGNN's unmodified `train_ASTGNN.py`, and then uses ASTGNN's own
`predict_main` to save predictions on the train and val sets as well as test. ASTGNN picks its own
best epoch and can't resume.

```bash
python astgnn.py prepare --dataset bikenyc               # prints one train command per channel
cd external/ASTGNN && python -u train_ASTGNN.py --config configurations/BikeNYC_paper_inflow.conf --cuda=0
python astgnn.py evaluate --dataset bikenyc              # combines the channels, scores with metrics.py
```

## Metrics

Predictions are clamped to `[0, 1]` and scored by `trafficpred/metrics.py`:

- **amount and placement**: `mse`, `mae` (scaled), `mse_frame` (summed per frame, the OpenSTL /
  PredFormer convention), `rmse_flow`, `mae_flow` (raw units)
- **where** (each map normalised to sum to 1): `sinkhorn` (distance-aware), `dist_mse`, `dist_tv`
  (fraction of traffic in the wrong cells)
- **how much** (each map summed): `volume_err` (relative), `volume_mae_flow`
- every metric per prediction step: `mse_t1`, `sinkhorn_t1`, ...

W&B (project `traffic-pred`, override with `WANDB_PROJECT`) gets live progress, val metrics every
epoch, prediction heatmaps, the final test metrics and their ratio to copy-last
(`vs_copy_last/*`, below 1 = better). Offline: `WANDB_MODE=offline`, then
`wandb sync wandb/offline-run-*`. Off: `--no-wandb`.

## Look at predictions

```bash
python visualize.py checkpoints/taxibj/distvol_sinkhorn.pt --browse
python visualize.py checkpoints/bikenyc/bikenyc_cnn_sinkhorn.pt --idx 50 --save figs/
```

or `notebooks/visualize.ipynb`. Checkpoints from the old training scripts can be rewritten into the
current format with `python tools/convert_checkpoints.py --verify checkpoints/*.pt`.
