# STDCformer on TaxiBJ

`train_stdcformer.py` trains STDCformer (He et al., arXiv:2412.02942) on TaxiBJ through `data.py`, using the same splits, `[0, 1]` scaling and split flags as the other trainers.

## Setup
Copy the `stdcformer/` folder and `train_stdcformer.py` into the repo root, next to `data.py`. No new dependencies are needed (torch, numpy, tqdm).

## Commands
```bash
python train_stdcformer.py --params_only                    # print size and split sizes, then exit
python train_stdcformer.py --amp                            # paper split + val, ~9.1M params
python train_stdcformer.py --amp --resume                   # continue after a kill / Slurm time limit
python train_stdcformer.py --d_model 64 --enc_layers 5      # the paper's own size (~1.1M)
python train_stdcformer.py --train-files 2013 2014 --val-files 2015 --test-files 2016
```

Output:
- `checkpoints/stdcformer.pt` holds the best weights by validation MAE.
- `stdcformer_last.pt` is written every epoch and is what `--resume` reads.
- `stdcformer_test.json` holds the test metrics.

## Parameter counts (32×32 grid, 1024 regions)

The number of heads doesn't change the count.

| `--d_model` | `--enc_layers` (= dec) | params |
|---|---|---|
| 64 | 5 | 1.13M (paper size) |
| **128** | **11** | **9.13M (default)** |
| 160 | 7 | ~9.2M |
| 192 | 5 | ~9.6M |

## How TaxiBJ maps onto the paper

| Paper | Here |
|---|---|
| Regions = NYC taxi zones | Regions = the 1024 grid cells |
| Zone adjacency → Laplacian eigenvectors | 8-neighbour grid graph (`--neighbors 4` for 4-neighbour) → Laplacian eigenvectors |
| Spatial confounders: zone ID, POI, housing, crime | Cell ID embedding + Laplacian eigenvectors |
| Temporal confounders: hour, weekday, holiday, weather | **Learned embedding of each step's position in the window**, because the windows from `data.py` carry no timestamps |
| 6 in / 6 out, 1-hour steps | `--t_in 4 --t_out 4`, half-hour steps |
| MAE loss | MAE by default (`--loss mse` to match the MSE-trained models) |

The temporal confounder is the main departure from the paper. The paper's back-door gate uses real time features (rush hour vs. night, weekday vs. weekend), and that is the "de-confounding" part. Position-in-window embeddings keep the architecture intact, but the gate can no longer tell 8am from 3am. The model accepts real time features (`tod_x/tod_y`, `dow_*`, `hol_*`, `exo_*` in the batch) if timestamps for each window can be added.

## Memory
Spatial attention compares every cell with every other cell (1024×1024 per head, per time step, per sample). On a CUDA GPU, PyTorch's memory-efficient and flash attention kernels avoid storing that matrix. On CPU or MPS it is stored in full, so batch size 8 with a tiny model already needed more than 7 GB on CPU. Train on a GPU (Sol), use `--amp`, and lower `--batch_size` if you hit out-of-memory errors.

## Metrics
Logged each epoch and saved for test:
- MSE and MAE on the `[0, 1]` scale
- MSE per future step
- RMSE and MAE in raw counts (using `scale_of(train)`)

These are not yet routed through `tracking.py`, so Sinkhorn, volume error and W&B logging are not included yet.
