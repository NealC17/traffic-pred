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
  losses.py               every training loss, independent of the models (see Losses below)
  sinkhorn.py             debiased Sinkhorn divergence on a grid, split_flow
  metrics.py              evaluate(): the metrics every model is scored with
  training.py             the shared training loop (Trainer)
  tracking.py             Weights & Biases logging
  checkpoints.py          load_model(path) / splits_for(checkpoint)
  astgnn.py               ASTGNN data conversion helpers
  viz.py                  heatmaps, interactive browser, notebook browser

tools/                    compress_data.py, import_grid.py, param_sizes.py, convert_checkpoints.py
notebooks/                visualize.ipynb
external/                 (gitignored) PredFormer, ASTGNN and STDCformer clones, ASTGNN experiments
datasets/                 compressed data
```

## Datasets

| `--dataset` | grid | step | default horizon | default split |
|---|---|---|---|---|
| `taxibj` | 2 × 32 × 32 | 30 min | 4 → 4 | test = last 4 weeks, val = 4 weeks before, train = rest |
| `bikenyc` | 2 × 21 × 12 | 1 h | 4 → 4 | test = last 10 days, val = 10 days before, train = rest |
| `nyctaxi` | 2 × 10 × 20 | 30 min | 4 → 4 | NYC taxi, Jan–Mar 2015 (STDN); test = last 20 days, val = 10 days before |
| `nycbike2` | 2 × 10 × 20 | 30 min | 4 → 4 | NYC Citi Bike, Jul–Aug 2016 (STDN); test = last 20 days, val = 10 days before |
| `taxibj21` | 2 × 32 × 32 | 30 min | 4 → 4 | 89 days of Beijing taxi flow, published already scaled to 0–1 (so "raw units" are that scale); test / val = 2 weeks each |
| `bousai_tokyo` | 2 × 80 × 80 | 30 min | 4 → 4 | Tokyo smartphone-GPS crowd flow, Apr–Jul 2017; test / val = 2 weeks each. Not included: see below |
| `bousai_osaka` | 2 × 60 × 60 | 30 min | 4 → 4 | the same for Osaka |
| `mmnist` | 1 × 64 × 64 | – | 10 → 10 | train generated on the fly, val a fixed generated set, test `mnist_test_seq.npy` |
| `synthetic` | 2 × 32 × 32 | – | 4 → 4 | drifting Gaussian hotspots, 70 / 15 / 15; a smoke test that needs no files |

Every dataset accepts `--t-in / --t-out` and `--train-size / --val-size / --test-size` (keep the
first N windows, or a fraction like `0.5`). File-based datasets can also be split by file:
`--train-files 2013 --val-files 2014 --test-files 2015 2016`. The min-max scale always comes from
the training data only. Windows never cross file or split boundaries.

`nyctaxi` / `nycbike2` come from [STDN](https://github.com/tangxianfeng/STDN) (Yao et al., AAAI'19,
`data.zip`, train and test files joined in time). STDN calls the two channels start and end volume.
The Bousai sets ([DeepCrowd](https://github.com/deepkashiwa20/DeepCrowd)) are given out by Yahoo!
Japan on application (https://randd.yahoo.co.jp/en/softwaredata); convert the `flowio_*.npy` files
they send with `tools/import_grid.py` (its `--help` has the commands). The same tool converts any
other (T, H, W, 2) or (T, 2, H, W) inflow/outflow array; register the result in
`trafficpred/data/datasets.py` and compress it with `python tools/compress_data.py <file.npy>`.

W&B: TaxiBJ runs log to project `traffic-pred`, every other dataset to `traffic-pred-<dataset>`
(e.g. `./train_all.sh --dataset nyctaxi` → `traffic-pred-nyctaxi`); `WANDB_PROJECT=...` overrides.

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

| `--model` | default loss | sinkhorn losses | defaults |
|---|---|---|---|
| `copy_last` | – | – | no training |
| `cnn`, `distvol` | sinkhorn | yes | 50 epochs, AdamW 1e-3, wd 1e-4, clip 1, one-cycle |
| `distvolformer` | sinkhorn | yes | same, but wd 1e-2 |
| `stresnet` | mse | yes | same; closeness branch only, output starts at the mean training value |
| `mlp` | mse | no | 50 epochs, wd 1e-2 |
| `predformer`, `predformer_official` | mse | no | 200 epochs (2000 on mmnist), wd 1e-2, paper configs |
| `stdcformer` | masked_mae | no | official config: 120 epochs, batch 64, Adam, wd 1e-4, clip 5, plateau, early stop 50 |

Any training flag (`--epochs`, `--lr`, `--batch-size`, `--weight-decay`, `--clip`, `--schedule`,
`--patience`) overrides the model's default.

### Losses

Losses live in `trafficpred/losses.py`, apart from the models: a model only names its default
loss and whether it predicts shape and volume separately. `mse`, `mae` and `masked_mae` work with
every model. The sinkhorn losses need a model that returns `x_hat = p_hat * n_hat`, with `p_hat` where the flow goes
(each map sums to 1) and `n_hat` how much there is (`cnn`, `distvol`, `distvolformer`, `stresnet`).
They are built from four terms: reconstruction `||x_hat - x||_1`, transport `S_eps(p_hat, p)`
(debiased Sinkhorn divergence), volume `|n_hat - n| / HW` and `KL(p || p_hat)` (the shapes compared
cell by cell, with no notion of distance between cells).

| `--loss` | trains |
|---|---|
| `sinkhorn` | `1 * reconstruction + 10 * transport + 1 * volume` |
| `sinkhorn_only` | transport alone (the volume head gets no gradient, so `n_hat` stays the last input frame's total) |
| `sinkhorn_vol` | `10 * transport + 1 * volume` |
| `kl` | `1 * reconstruction + 0.1 * KL + 1 * volume`: KL in place of Sinkhorn |
| `sinkhorn_kl` | `1 * reconstruction + 10 * transport + 1 * volume + 0.1 * KL` |
| `sinkhorn_change` | `1 * reconstruction + 1 * volume + 0.3 * change`: Sinkhorn between where the flow grows (and shrinks) relative to the last input frame in the prediction and in truth, each normalized to a distribution, maps weighted by their true change; sharp and converged (`--eps 0.002` ≈ 1.4-cell blur, 60 iterations with epsilon scaling) |
| `sinkhorn_vol_uw` | the vector (transport, volume), combined with learned uncertainty weights (Kendall et al. 2018): `sum_i exp(-s_i) L_i + s_i`. `s_i` starts at `log L_i` of the first batch, is trained without weight decay and is saved in the checkpoint; W&B logs the weights as `epoch/train_weight_sinkhorn` / `_volume` |
| `sinkhorn_adaptive` | all three terms, reweighted like Adam: every 10 steps (`--adapt-every`) each term's gradient on the model's parameters is measured, running averages of the squared norms (`--adapt-beta 0.95`, bias-corrected) set `w_i = sqrt(v_l1 / v_i)`, so each term pushes as hard as the L1 term; `--l1 / --l2 / --l3` scale those shares (default 1 each). The averages are saved for `--resume`; W&B logs `epoch/train_weight_*` |

`--l1 / --l2 / --l3 / --l4 / --l5` change the weights (reconstruction / transport / volume / KL /
change) of the fixed-weight losses, `--sinkhorn-anneal` turns on epsilon scaling in any Sinkhorn
loss (needed for `--eps` below ~0.005 to converge), `--eps / --sinkhorn-iters`
the Sinkhorn settings of any of them. A value that differs from the loss's default is added to
the run name (`--loss sinkhorn --l1 0` -> `distvol_sinkhorn_l1-0`), so it never overwrites the
default run. Every term is logged whether or not it is trained (Sinkhorn only when its weight
is above 0). At the default weights the terms are far apart in size: late in training 10 * Sinkhorn
is about 3% of the loss on TaxiBJ, while a copy-last prediction has KL ≈ 2x its L1, hence KL's 0.1.

Why `sinkhorn_change`: the whole-city shape hardly moves between frames and is mostly static
background, and at `eps 0.01` (≈ 3-cell blur) Sinkhorn charges a 1-cell blur of the map ~50x less
than a 1-cell shift. So the `transport` term barely sees the errors models make: the `sinkhorn`
runs end with no lower test Sinkhorn than the same loss without it (`--l2 0`). The change term
looks only at the moving part, at full scale (on a trained distvolformer it is ~30% of L1 with
2.4x L1's gradient). Below `eps` ≈ 0.006 the log-domain solver is `torch.compile`d on CUDA
(~12x faster; `TRAFFICPRED_NO_COMPILE=1` turns it off). `./train_all.sh distvol_` and
`./train_all.sh distvolformer_` run each of those models with mse and all four sinkhorn losses.

Protocol: every epoch is scored on val, the checkpoint with the lowest val MSE (`--select`) is
kept, and at the end that checkpoint is scored on train, val and test (MSE, RMSE and MAE in raw
units, plus every metric below). The seed defaults to 42. Checkpoints go to
`checkpoints/<dataset>/<run>.pt` and hold the weights, everything needed to rebuild the model and
its data split, and the val / test metrics. `<run>.last.pt` is written every epoch for `--resume`,
which refuses to continue a run made with different settings.

### Tuning the loss weights

`tune_loss.py` searches loss weights with Optuna (TPE). `--space kl` (default, study
`<model>_<dataset>`) searches `sinkhorn_kl`; `--space change` (study `<model>_<dataset>_change`)
searches `sinkhorn_change` (`--l2` Sinkhorn or off, `--l3`, `--l5` change or off, `--eps`
0.0005-0.01) and also queues `1 L1 + 0.034 volume`, the kl study's best plain weighting, as a
control. In the kl space l1 stays 1 and it picks
`--l2` (Sinkhorn, 0.3-1000, or off), `--l3` (volume, 0.03-30), `--l4` (KL, 0.003-3, or off) and
`--eps` (0.003-0.1), so it covers `sinkhorn`, `kl` and plain L1 + volume as special cases. Trial 0
is the default weighting. Each trial is a shortened run (`--epochs 20`, one seed, the one-cycle
schedule squeezed to fit, no checkpoints, `--curves val`) scored by its best **val** RMSE, and a
trial behind the median of the earlier ones at the same epoch is pruned. Every trial is a W&B run
`hyperparameter_tuning_<study>_t<N>` in group `hyperparameter_tuning_<study>`, tagged
`hyperparameter_tuning`, in its own W&B project `traffic-pred-hyperparameter-tuning`
(`--wandb-project`), so trials never mix with the training runs in `traffic-pred`.

On one machine, `./tuning.sh` runs it in the background (default `--until 40`, log
`logs/hyperparameter_tuning_<study>.out`) and won't start a second copy while one is running:

```bash
./tuning.sh                  # start, or continue, the study (options go to tune_loss.py)
./tuning.sh status           # running? best trials so far
./tuning.sh log              # follow the log
./tuning.sh stop             # stop; the current trial is queued again
./tuning.sh confirm 3        # -> train_tuned_<study>.sh
```

```bash
python tune_loss.py --trials 40                          # study distvolformer_taxibj
python tune_loss.py --study dvf_l --trials 40 -- --model distvolformer --dim 256 --heads 8
python tune_loss.py --space change --until 40            # sinkhorn_change: l2, l3, l5, eps
python tune_loss.py --fix l4=0 --study no_kl --trials 30 # hold a weight fixed
python tune_loss.py --report                             # best trials, parameter importances
python tune_loss.py --confirm 3                          # -> train_tuned_<study>.sh
sbatch --array=0-3 --gres=gpu:1 --time=08:00:00 --mem=32G --cpus-per-task=4 \
       --wrap="python tune_loss.py --trials 10"          # 4 workers sharing one study
```

A study is one journal file, `checkpoints/tuning/<study>.journal`. Rerunning the same command
continues it, and several workers can share it. A trial a killed worker left running (no progress
for `--stale-minutes 30`) is failed and its weights are queued again. A study's options (train.py
arguments, epochs, seed, `--fix`) can't change; use a new `--study` instead. Single-seed 20-epoch
runs are only a screen: `--confirm K` writes `train_tuned_<study>.sh`, a `train_all.sh` wrapper that
runs the K best weightings and the default `sinkhorn` at full length on seeds 42, 7, 123 and 2025
(workspace `tuned_<study>`).

`slurm/seeds_then_tune.sbatch` chains the two on Sol: it finishes `./train_all_new_seeds.sh`
(every run with a `.done` marker), then runs `tune_loss.py --until 40` and `--confirm 3`. 15 min
before each job's time limit it stops, resubmits itself to start after it ends (an interrupted
trial is queued again), so it carries on across as many jobs as needed. How to submit it after a
running job without touching that job's code is in the script's header.

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

On `nyctaxi` / `nycbike2`, `--external` gives `cnn`, `distvol` and `distvolformer` the same 10 calendar
features (the day-off flag also covers public holidays), and `--weather` adds 6 NYC weather features
from Open-Meteo (`datasets/weather/*.json`: temperature, wind, log precipitation, log snowfall, log
snow depth, cloud cover). Each frame gets the hourly record at or before its end, and the targets
get the last input frame's weather. There are three ways in for DistVol / DistVolFormer:

- default (conditioning): the target steps' features go through a zero-initialised MLP that is
  added to the volume code, so they reach the FiLM / AdaLN of the map branch and the volume head
  (run names `_ext` / `_extw`)
- `--ext-input`: every frame's features (input frames, day / week-back frames and targets) go
  through a shared 8-dim embedding and into the encoders as constant input maps; with
  `--time-tokens` they go per frame, and onto the target tokens (`_extin` / `_extwin`)
- `--ext-volume`: like the default, but the MLP's output reaches the volume head only, so the map
  branch never sees it (`_extvol` / `_extwvol`)

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
./train_all.sh --dataset bikenyc stdcformer astgnn # any registered dataset: taxibj (default), bikenyc,
                                                   # mmnist, synthetic; some runs only
./train_all.sh --quick                             # 1 epoch each: pipeline check
./train_all.sh --no_save --quick                   # same check, but saves no checkpoints
./train_all.sh --batch-size stdcformer=8           # one model's batch size (repeatable; N alone = all)
./train_all.sh --workspaces                        # list workspaces
./train_all.sh --help
./train_all.sh --seeds 42 7 123                    # every run once per seed (names get _seedN, except 42)
```

`./train_all_new_seeds.sh` is a seed and model-size study of the DistVol / DistVolFormer losses on top of
`train_all.sh` (same options): mse, sinkhorn, sinkhorn `--l2 0` (no Sinkhorn term), sinkhorn_adaptive, kl
and sinkhorn_kl, at
the default sizes (3.95M / 7.53M params) and at PredFormer's size (`distvol_l`: width 96, 8.72M;
`distvolformer_l`: dim 256, 8 heads, 9.94M), each with seeds 42, 7, 123 and 2025. It uses its own
workspace, `new_seeds`, and continues it when rerun; it doesn't change which workspace `./train_all.sh`
continues. Wrappers like it set `RUNS` (lines may add `train.py` arguments), `SEEDS` and
`DEFAULT_WORKSPACE`, then source `train_all.sh`.

On `mmnist` (64x64 = 4096 cells) `train_all.sh` skips STDCformer unless you give it a batch size
(`--batch-size stdcformer=1`): its attention over every cell needs about 40 GB at batch 8. Moving MNIST is fixed
at 10 in -> 10 out and has no file split, and PredFormer uses its paper's 2000 epochs there.

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

For the STDN-protocol runs, `tools/compare_published.py` scores runs against the papers' numbers.
`tools/risk_eval.py` gives an empirical risk analysis:
- the train / val / test RMSE and the generalisation gap;
- tail risk (per-window percentiles and CVaR95);
- RMSE by demand quartile;
- a day-level block bootstrap of each run's difference from a `--baseline` run.

W&B (project `traffic-pred`, override with `WANDB_PROJECT`) gets live progress, val metrics every
epoch, prediction heatmaps, the final test metrics and their ratio to copy-last
(`vs_copy_last/*`, below 1 = better). Offline: `WANDB_MODE=offline`, then
`wandb sync wandb/offline-run-*`. Off: `--no-wandb`.

## Experiments and results

Every experiment run so far, roughly in order. Unless noted:

- **Protocol:** STDN's, i.e. next step, RMSE on cells whose true count is above 10
  (`--threshold 10.5`), per channel Start / End.
- **Seeds and settings:** mean of 3 seeds (42, 7, 123), untuned defaults.
- **Scoring:** `tools/compare_published.py` (says BETTER when the seed mean + 2 standard errors is
  below the target) and `tools/eval_published.py`.
- **Old threshold:** runs before 2026-11-21 kept cells equal to 10 (≥ 10). That makes their NYCTaxi
  numbers about 0.3 too good; NYCBike2 is unaffected. Those numbers are marked (≥10).

Published targets (RMSE Start / End):

| | STDN (AAAI'19) | Wang et al. 2020, no outside data | Wang et al. 2020, with weather / holidays |
|---|---|---|---|
| NYCTaxi | 24.10 / 19.05 | 23.69 / 18.78 | 22.37 / 17.78 |
| NYCBike2 | 8.85 / 8.15 | – | 8.32 / 7.66 |

Wang et al. is arXiv 2005.05128, an unnamed extension of STDN.

### Headline

| Dataset / protocol | Our best fair result | Published | Verdict |
|---|---|---|---|
| NYCTaxi, STDN | DistVolFormer, fold-val (#25): **22.35 / 18.01** | Wang + weather 22.37 / 17.78 | Beats STDN and Wang without outside data; ties Wang with weather on Start, behind on End, using no outside data |
| NYCTaxi, STDN (no fold-val) | DistVolFormer, tuned (#15): 23.24 / 18.59 | STDN 24.10 / 19.05 | Beats STDN and Wang without outside data |
| NYCBike2, STDN | DistVolFormer, fold-val (#25): **8.34 / 7.66** | Wang + weather 8.32 / 7.66 | Beats STDN; ties Wang with weather, using no outside data |
| NYCTaxi, ST-SSL (MAE, true > 5) | DistVolFormer (#20): **11.84 / 9.49** | ST-SSL 11.99 / 9.78 | Beats |
| NYCBike2, ST-SSL | DistVolFormer (#20): 5.08 / 4.82 | ST-SSL 5.04 / 4.71 | Not beaten |
| BikeNYC, DeepSTN+ (all cells) | DistVolFormer (#10): RMSE 5.93, MAE 2.14 | DeepSTN+ 5.955 / 2.285 | RMSE ties, MAE beats; ST-ResNet's 6.475 beaten easily |
| TaxiBJ, next step | DistVolFormer 4 → 4 scored at t+1, 1 seed (#1): **16.34** | ST-ResNet 16.69, ACFM 15.40 | Beats ST-ResNet, not ACFM |

What held up:
- the DistVol shape × total factorization;
- L1 + volume instead of MSE;
- day / week-back frames;
- tuning;
- training on the most recent data (fold-val).

Every OT term, attention variant, head and learned cost, the frame encoder, and calendar and
weather in every form gave no RMSE gain or hurt. OT only helps the spatial (GeOT) error.

### TaxiBJ

1. **Baseline, 4 → 4.** Train 2013, val 2014, test 2015–16; RMSE over all frames.
   - DistVolFormer + Sinkhorn is best at 18.85. Then DistVolFormer MSE 19.23, DistVol 19.25, CNN
     19.61, ST-ResNet 19.77, MLP 20.67, copy-last 38.4.
   - Sinkhorn without a count-space loss breaks (44–65).
   - Scored at t+1: 16.34 vs ST-ResNet's published 16.69 (1 seed, no outside data).
2. **Optuna on DistVolFormer's loss weights.** It drives the Sinkhorn weight to about 0. The best
   is L1 + volume, 17.21 vs 17.40 for the default (short runs, 1 seed).
3. **Official PredFormer, OpenSTL protocol.** Frame MSE 0.42 vs the paper's 0.277–0.284. Not
   reproduced, and the gap is unexplained.
4. **Next-step TaxiBJ** (`train_taxibj_nextstep.sh`, Slurm). Runs on a separate machine; no results
   here yet.

### NYCTaxi / BikeNYC, early 4 → 4 (all-cell MSE; not comparable to papers)

5. **Inputs.**
   - A residual-over-the-last-frame output hurts (about +5 MSE).
   - Day / week-back frames help a lot on BikeNYC (DistVolFormer 48 → 39) and are neutral on
     NYCTaxi.
   - Sinkhorn with `--l2 0` is best, and that is L1 + volume with no transport term.
6. **Transport head.** No gain (196–208 vs 197).
7. **Sinkformer attention.** No gain.
8. **MSE vs Sinkhorn, 4 seeds.** DistVolFormer: MSE 201, Sinkhorn 204.7, L1 + volume 197.2. The OT
   term adds nothing.
9. **Longer horizons (8 and 12 steps).** OT hurts NYCTaxi more as the horizon grows (+0.4 at t+1 →
   +1.0 at t+8). Neutral on bikes.

### NYCTaxi / NYCBike2, STDN protocol

10. **Published sweep, 4 frames in.** Both of these beat STDN:
    - NYCTaxi (≥10): ST-ResNet + day / week-back 23.05 / 18.12, DistVolFormer 23.45 / 18.88.
    - NYCBike2: ST-ResNet + day / week-back 8.69 / 7.90.
    - BikeNYC is in the headline.
11. **4, 8 or 12 input frames.** 8 is best for taxi, 4 for bikes.
    - ST-ResNet + day / week-back: taxi 22.89 / 18.11 (≥10), bike 8.69 / 7.91.
    - DistVolFormer: taxi 23.15 / 18.64 (≥10), bike 8.94 / 8.18.
12. **Day / week-back variants and blends.**
    - A shifted day / week-back window helps ST-ResNet: taxi 22.67 / 17.92 (≥10).
    - A 9-model blend gets taxi 21.63 / 17.14 (≥10) and bike 8.34 / 7.63. That is an ensemble, so
      not a fair comparison.
13. **Time tokens.** A small gain: taxi 23.04 / 18.38 (≥10), bike 8.73 / 8.17 (patch 1).
14. **Loss ablation** (MSE, L1, Huber, L1 + volume, Sinkhorn, with and without the level term).
    - The L1 family is slightly better than MSE: taxi DistVol L1 + volume 23.06 / 18.49 (≥10).
    - Sinkhorn is DistVolFormer's worst loss.
15. **Optuna model tuning.**
    - The first pass made ST-ResNet the taxi leader.
    - The second pass, tuned for the ≥ 10 cells: DistVolFormer 22.90 / 18.28 (≥10), which is
      23.24 / 18.59 under > 10. Bike 8.66 / 8.13.
    - The graph variant gets bike 8.60 / 8.01.
    - The official PredFormer, our PredFormer and the MLP are far behind.
16. **Loss focus.**
    - No extra weight on low-count cells is a small gain: DistVolFormer taxi 22.82 / 18.25 (≥10),
      bike 8.57 / 8.09.
    - Training on log counts is worse.
17. **Calendar features, ST-ResNet.** No gain (taxi 23.85 → 23.95).
18. **Direct output, without the shape × total factorization.** Much worse: taxi 24.94 / 19.39 vs
    22.90 / 18.28. The factorization is a real win.
19. **Split loss** (OT on the map, separate volume loss). Far worse: taxi 29–33, bike 11–13.
20. **ST-SSL protocol** (MAE, true > 5). See the headline. On bikes we beat AGCRN and STGCN.
21. **OD trip flows as extra input** (outside data, so not comparable to the papers).
    - Taxi: helps DistVol (21.06 → 20.58 all-cell) and the CNN, but the best is still DistVolFormer
      without OD.
    - Bike, with the OD head: DistVolFormer 8.30, the graph variant 8.24 (End 7.77).
22. **OD per time-of-day kernel, and retuning with OD.** Neither helps; the retunes are worse.
23. **Usual-pattern (weekly average) input.**
    - No gain for DistVolFormer.
    - A big gain for the CNN (bike 8.62 → 8.38).
    - Hurts ST-ResNet.
24. **Diagnostics.**
    - Error split: giving the model the true total barely moves RMSE (about 20.8 → 19.2), while
      the true map drops it to about 8. The error is in the map, not the total.
    - A ratio-to-usual feature could remove 4–7% of MSE after the fact, but didn't carry over into
      training.
    - Correcting the level with recent data helps every model. That points to drift, which led to
      #25.
25. **Fold-val** (train on train + val, keep the last epoch). **Our best result.**
    - DistVolFormer: taxi 22.35 / 18.01, bike 8.34 / 7.66.
    - CNN: bike 8.53 / 7.57.
    - ST-ResNet diverges without a validation set to pick its epoch.
    - Caveat: the configs were tuned on the validation set before it was folded in.
26. **Unbalanced OT and the change (ΔX±) term, 4 seeds.**
    - Unbalanced OT is worse everywhere.
    - The change term ties L1 + volume and gives the CNN about 2%.
    - GeOT spatial error (`tools/geot_eval.py`): L1 cuts it 20–28% vs MSE, and OT terms add another
      1–7% at a small RMSE cost. Bike: DistVolFormer unbalanced 112 vs 116 for L1 + volume.
27. **OT diagnostics.**
    - At small blur the OT plan is close to the identity (87% of mass stays put).
    - A cost learned from OD flows is no better than grid distance.
    - Neighbouring cells err in the same direction, so the error is level, not displaced mass.
    - Timing shifts explain nothing.
28. **Frame encoder** (`--frame-encoder`). Worse than stacking frames (taxi 25.01 / 19.59 vs
    23.97 / 18.91) and worse than STDN.
29. **Calendar and weather as conditioning** (`--external [--weather]`, see "External data").
    - Weather hurts every taxi model: DistVol 23.38 → 24.51, CNN 25.28 → 29.53.
    - Bikes and calendar are neutral.
    - The best fair taxi result here is DistVol with no outside data, 23.38 / 18.77. It beats STDN
      and Wang without outside data on Start.
30. **Log counts as input.** Taxi worse by 0.15–0.3; bike neutral.
31. **Calendar and weather as encoder input** (`--ext-input`). No gain, and weather is worse than
    as conditioning:

    | | none | calendar, conditioning | calendar, input | weather, conditioning | weather, input |
    |---|---|---|---|---|---|
    | NYCTaxi DistVol | 23.38 / 18.77 | 23.42 / 18.87 | 23.47 / 18.85 | 24.51 / 19.77 | 25.77 / 20.84 |
    | NYCTaxi DistVolFormer | 23.47 / 18.99 | 23.44 / 19.07 | 23.39 / 18.92 | 23.82 / 19.45 | 24.95 / 20.67 |
    | NYCBike2 DistVol | 9.05 / 8.60 | 9.02 / 8.45 | 9.01 / 8.39 | 8.92 / 8.34 | 9.13 / 8.53 |
    | NYCBike2 DistVolFormer | 8.93 / 8.18 | 8.93 / 8.30 | 8.87 / 8.24 | 9.07 / 8.33 | 8.90 / 8.22 |

    The NYCTaxi test period has different weather from training:
    - mean temperature −8.7 °C vs −3.7 °C, with a low below anything seen in training;
    - no hours of heavy precipitation vs 24;
    - snow on the ground the whole time.

    NYCBike2's test weather looks like its training weather. So weather inputs fit a few events in
    training that don't recur at test time. Month or week number wouldn't help either: the 60-day
    datasets never show the test period's values in training.
32. **Weather on the volume head only** (`--ext-volume`). This is the least harmful way in for
    weather, but it is still no gain on NYCTaxi. On NYCBike2 the small edge for DistVolFormer is
    within seed noise:

    | | none | weather, conditioning | weather, input | weather, volume head only |
    |---|---|---|---|---|
    | NYCTaxi DistVol | 23.38 / 18.77 | 24.51 / 19.77 | 25.77 / 20.84 | 24.13 / 19.23 |
    | NYCTaxi DistVolFormer | 23.47 / 18.99 | 23.82 / 19.45 | 24.95 / 20.67 | 23.65 / 19.03 |
    | NYCBike2 DistVol | 9.05 / 8.60 | 8.92 / 8.34 | 9.13 / 8.53 | 9.00 / 8.39 |
    | NYCBike2 DistVolFormer | 8.93 / 8.18 | 9.07 / 8.33 | 8.90 / 8.22 | 8.79 / 8.16 |

    Keeping weather out of the map branch removes most of the taxi damage, so most of that damage
    came from the map branch. The total forecast doesn't use weather well either: it is still
    slightly worse than no weather.
33. **Spatial modes, experiments A0 and A** (`train_modes.sh`; baseline DistVolFormer with time
    tokens, 23.47 / 18.99 and 8.93 / 8.18). The motivating checks, on train data:
    - The maps are low-rank: 16 modes explain 93% of NYCTaxi's maps.
    - The step-to-step changes are much less so: 32 modes capture 93% on NYCTaxi and only 53% on
      TaxiBJ.
    - Modes fit on the training period capture about 10 points less of the test period's changes
      at r = 8–16. The modes drift.

    Variants:
    - A0, `--mode-head R`: the logits get + Φa, with Φ starting from the train split's change modes
      (PCA) or at random.
    - A, `--spatial-modes R`: the spatial blocks mix tokens through R fixed learned modes,
      U(VᵀH)W, instead of attention. 7.18M parameters vs 7.51M.

    | | NYCTaxi | NYCBike2 |
    |---|---|---|
    | attention (baseline) | 23.47 / 18.99 | 8.93 / 8.18 |
    | A, modes r = 8 | 23.44 / 18.90 | 8.97 / 8.37 |
    | A, modes r = 16 | 23.44 / 18.80 | 9.08 / 8.41 |
    | A, modes r = 32 | 23.56 / 18.90 | 9.08 / 8.34 |
    | A0, head, random, r = 32 | 23.59 / 18.94 | 9.34 / 8.64 |
    | A0, head, PCA, r = 16 | 23.74 / 19.09 | 9.15 / 8.56 |
    | A0, head, PCA, r = 32 | 23.75 / 19.03 | 9.27 / 8.76 |

    - A0 is worse everywhere, and the PCA start is worse than the random one on NYCTaxi. A low-rank
      term on top of a full-rank output adds nothing, and the train-period modes drift.
    - A is mixed. On NYCTaxi, fixed modes tie attention on Start and are slightly better on End (r =
      16: 18.80 ± 0.03 vs 18.99 ± 0.11, about 3 standard errors). On NYCBike2 they are worse on End
      by about 0.2 (Start is within noise).
    - So fixed, input-independent spatial mixing is about as good as attention on taxi but not on
      bikes. Input-dependent gating (experiment B) is the next test.
    - Risk analysis on NYCTaxi (`tools/risk_eval.py`, both channels together):
      - Every variant has the same generalisation gap, train 17.0–17.3 → test 21.0–21.1 (about +3.9).
        The mixer fits the training period slightly better without overfitting more.
      - The worst 5% of windows (CVaR95 about 35.6–36.0) are hit about equally.
      - The mixer's edge is mostly in the quietest quartile of windows (13.45 vs 13.70).
      - Against attention, r = 16 is −0.11 RMSE, with a 95% interval of [−0.29, +0.05] from a
        bootstrap over the 20 test days, and better in 91% of resamples. Not established. The
        test-day uncertainty (about ±0.17) is larger than the seed spread that `compare_published`'s
        verdicts use.
34. **Gated spatial modes, experiment B** (`train_modes_gated.sh`, `--spatial-modes R --mode-gate`).
    Each mode is scaled by 1 + tanh(G(Z)), with G linear within and across modes and zero at the
    start. Worse than fixed modes on both datasets:

    | | NYCTaxi | NYCBike2 | NYCBike2 vs attention, both channels [95% day-bootstrap] |
    |---|---|---|---|
    | attention | 23.47 / 18.99 | 8.93 / 8.18 | – |
    | A, fixed, r = 8 | 23.44 / 18.90 | 8.97 / 8.37 | +0.12 [+0.04, +0.20] |
    | A, fixed, r = 16 | 23.44 / 18.80 | 9.08 / 8.41 | +0.19 [+0.13, +0.26] |
    | B, gated, r = 8 | 23.57 / 18.95 | 9.15 / 8.55 | +0.29 [+0.22, +0.37] |
    | B, gated, r = 16 | 23.51 / 18.97 | 9.19 / 8.48 | +0.28 [+0.19, +0.39] |

    - On NYCTaxi, gating loses A's small edge and lands on attention (+0.01 and +0.03 against it).
    - On NYCBike2 every mode variant is worse than attention, with day-bootstrap intervals that
      exclude zero, and gating makes it worse still.
    - The gated r = 16 model has a higher bike train error (7.52 vs 6.79 for attention), so it is
      harder to train rather than overfitting.
    - Conclusion: making the modes depend on the input doesn't recover what attention does on
      bikes. This doesn't support the dynamic-modes direction (experiment C).
35. **OT destination kernel from counts alone** (NYCBike2, `tools/ot_od_kernel.py`, `--od-kernel`).
    Entropic OT between the mean Start map at slot s and the mean End map at s+1 is the doubly
    constrained gravity model. Its plan estimates where trips go, with no OD data. Measured against
    the real per-slot kernel from STDN's trips, using training days only:

    | kernel | TV to real | mean trip (cells) | arrivals error |
    |---|---|---|---|
    | real all-day kernel (OD data) | 0.224 | 2.43 | 27.5% |
    | Sinkhorn OT, ε = 2 | 0.270 | 2.22 | 11.6% |
    | gravity without OT balancing, ε = 2 | 0.284 | 2.18 | 27.7% |
    | destination share only | 0.358 | 3.47 | 16.0% |
    | identity | 0.939 | 0 | 53.2% |

    - The OT kernel's expected arrivals are closer to the real slot kernel's than the real all-day
      kernel's are, and the OT balancing step halves the error. The real per-slot kernel's mean trip
      is 2.45 cells.
    - Training check (`train_od_kernel.sh`): the tuned DistVolFormer with `--od --od-slot --od-head`,
      with the real kernel swapped for the OT one or the destination-share one (both the head's prior
      and the arrivals input). The in-progress input map is still from the OD data. STDN metric
      (test RMSE, cells with true > 10), mean ± sd over seeds 42 / 7 / 123:

      | kernel | Start | End | seed ensemble S / E |
      |---|---|---|---|
      | real per-slot kernel | 8.62 ± 0.06 | 7.96 ± 0.06 | 8.48 / 7.87 |
      | Sinkhorn OT, ε = 2 | 8.63 ± 0.08 | 7.98 ± 0.06 | 8.49 / 7.90 |
      | destination share only | 8.64 ± 0.04 | 8.09 ± 0.12 | 8.49 / 7.99 |
      | no OD (tuned DistVolFormer) | 8.66 ± 0.02 | 8.13 ± 0.04 | 8.52 / 8.01 |

    - The OT kernel recovers almost all of the real kernel's gain (End 7.98 vs 7.96), and the
      destination-share kernel only part of it. All three beat STDN (8.85 / 8.15) and trail Wang et
      al. with weather (8.32 / 7.66). OT is closed after this.

## Look at predictions

```bash
python visualize.py checkpoints/taxibj/distvol_sinkhorn.pt --browse
python visualize.py checkpoints/bikenyc/bikenyc_cnn_sinkhorn.pt --idx 50 --save figs/
```

or `notebooks/visualize.ipynb`. Checkpoints from the old training scripts can be rewritten into the
current format with `python tools/convert_checkpoints.py --verify checkpoints/*.pt`.
