# PriMoTraj

Minimal runnable code for **PriMoTraj: Motion-Prior Gating for Lightweight
GPS-Derived Trajectory Forecasting**.

PriMoTraj predicts short-horizon GPS trajectories from recent coordinates alone
— no road graph, POI features, user histories, or pretraining. A compact
temporal encoder drives a sample-wise softmax gate over seven deterministic
motion priors, and a bounded residual head corrects the gated prior, so the
coordinate output stays tied to an explicit extrapolation.

The deployed configuration is **11,129 parameters** at `H=6` and **11,993** at
`H=12` (2,247 of them in the gate). `python verify_model_config.py` prints these
counts and checks the prior algebra.

## Quick check (2 minutes, no dataset needed)

```bash
pip install -r requirements.txt
python verify_model_config.py                 # architecture + parameter counts
python make_demo_data.py --output data/demo.npz
DATA=data/demo.npz SEEDS=2024 PRED_LENS=6 EPOCHS=1 BATCH=128 USE_GPU=0 \
  bash run_porto_main.sh
python collect_results.py logs
```

`make_demo_data.py` writes synthetic curves in the Porto array format. It only
proves the code runs — the numbers it produces are not results.

## Real experiments

### 1. Data

Datasets are not included. Preprocess them into the `.npz` format the loaders
expect:

```bash
# Porto taxi (Kaggle "Taxi Trajectory Prediction", train.csv)
python primotraj/utils/preprocess_porto.py \
  --input data/train.csv --output data/porto_processed.npz

# Fixed-interval variants used in the paper (15s / 30s / 60s / 120s / 300s).
# The main Porto table uses porto_ds15.npz.
python primotraj/utils/downsample_porto_npz.py \
  --input data/porto_processed.npz --out_dir data \
  --factors 1 2 4 8 20 --base_interval_sec 15 --min_len 18 \
  --report_seq_len 12 --report_pred_lens 3 6 12 24

# T-Drive (sparse regime) and Argoverse 2 (high-frequency support)
python primotraj/utils/preprocess_tdrive.py --output data/tdrive_processed_5min.npz
python primotraj/utils/preprocess_av2.py --input_root <av2_motion_forecasting_root>
```

Splits are trip-level 70/10/20, assigned inside the loader.

### 2. Main Porto comparison

`T=12`, `H` in `{6, 12}`, 3 seeds, for PriMoTraj and every baseline shipped here:

```bash
DATA=data/porto_ds15.npz bash run_porto_main.sh
python collect_results.py logs --csv results.csv
```

`collect_results.py` parses the `BENCHMARK|` line each run prints and reports
mean +/- std over seeds, meter-level ADE/FDE (Haversine, computed after inverse
normalization), parameter count and batch inference time.

### 3. A single PriMoTraj run

```bash
python -u primotraj/main_traj.py \
  --seed 2024 --data data/porto_ds15.npz \
  --checkpoint_dir checkpoints/PriMoTraj --name seed2024_pred6 \
  --seq_len 12 --pred_len 6 \
  --train_epochs 5 --batch_size 2048 --learning_rate 5e-5 \
  --use_mdm True --use_moe False \
  --motion_prior cvmix --motion_prior_weight 1.0 \
  --motion_prior_mode gate --motion_prior_recent_weight 0.5 \
  --motion_prior_damping 0.8 \
  --motion_prior_n_priors 7 \
  --motion_prior_gate_hidden 64 --motion_prior_gate_horizon 3 \
  --motion_prior_residual_head full_window \
  --motion_prior_residual_hidden 64 --motion_prior_residual_init 0.15 \
  --mse_loss_weight 0.1 --fde_loss_weight 1.0 --ade_loss_weight 0.0 \
  --weight_decay 1e-4 --grad_clip_norm 1.0 --min_learning_rate 1e-6
```

Evaluate a saved checkpoint with `python primotraj/load_best.py --checkpoint
<path>/best.pt --data <data>.npz`, passing the same hyperparameters used at
training time (checkpoints hold weights only).

## What is where

| Path | Contents |
|---|---|
| `primotraj/models/tsAMD.py` | The model: temporal encoder, motion-prior bank, gate, residual head |
| `primotraj/models/common.py` | RevIN, Parallel Mixer (PM), Multi-scale Decomposable Mixing (MDM) |
| `primotraj/main_traj.py` | Training / evaluation entry point |
| `primotraj/utils/` | Dataset preprocessing, sliding-window loader, seeding |
| `baselines/physics_baselines.py` | Zero-parameter baselines: Last, CV, CVMix, CV-KF |
| `baselines/traj_utils.py` | Shared raw-window loader, meter-level evaluator, timing |
| `GRU/main_traj.py` | The recurrent baseline; `--rnn_type LSTM` is the LSTM-Seq2Seq-Attn row |
| `DLinear/main_traj.py` | DLinear baseline |
| `third_party_adapters/` | Adapters and patches for PatchTST / TimeMixer / iTransformer, which must be cloned from upstream (see its README) |
| `run_porto_main.sh` | Runs PriMoTraj and all in-package baselines over seeds and horizons |
| `collect_results.py` | Aggregates run logs into the comparison table |
| `verify_model_config.py` | Checks parameter counts and the prior bank |
| `analysis/` | Per-window error dumps, the trip-level paired bootstrap, the gate-weight dump, and the ADE/parameter trade-off plot |

## Model options

| Flag | Deployed value | Meaning |
|---|---|---|
| `--motion_prior_n_priors` | `7` | Prior bank: constant-position, constant-velocity, smoothed, mixed, damped, constant-acceleration, long-window smoothed |
| `--motion_prior_gate_hidden` | `64` | Gate hidden width `h_g` |
| `--motion_prior_gate_horizon` | `3` | `K_g`: trailing input steps the gate reads |
| `--motion_prior_residual_head` | `full_window` | Residual MLP over the whole input window with a learned per-step bound |
| `--motion_prior_residual_hidden` | `64` | Residual hidden width `r` |
| `--motion_prior_residual_init` | `0.15` | Initial value of the per-step bound |
| `--motion_prior_recent_weight` | `0.5` | Mixing coefficient of the mixed-velocity prior |
| `--motion_prior_damping` | `0.8` | Damping coefficient of the damped prior |

Setting `--motion_prior_n_priors 5 --motion_prior_gate_hidden 16
--motion_prior_residual_head none --motion_prior_residual_scale 0.02` selects the
earlier compact configuration (1,653 parameters at `H=6`), which
`verify_model_config.py` also checks.

## Analysis scripts

`analysis/` holds the scripts behind the statistical tables and the trade-off
figure:

```bash
# Per-window ADE/FDE on the 131,072-window matched test subset, plus the trip id
# of every retained window.
python analysis/compute_sample_errors_v2.py --model_type primotraj \
  --checkpoint checkpoints/PriMoTraj/seed2026_pred6/best.pt \
  --data_path data/porto_ds15.npz --pred_len 6 --use_gpu \
  --output errors/PriMoTraj_p6.npz

# Same for a zero-parameter comparator (no checkpoint needed).
python analysis/compute_sample_errors_v2.py --model_type cv_kf \
  --data_path data/porto_ds15.npz --pred_len 6 --output errors/CV-KF_p6.npz

# Trip-level paired bootstrap: whole trips are resampled with replacement, so
# windows from one trip stay together. Reports the observed tail count and a 95%
# confidence interval on each delta at whatever budget you pass.
python analysis/make_paired_bootstrap_v2.py --target errors/PriMoTraj_p6.npz \
  --baseline CV-KF=errors/CV-KF_p6.npz --pred_len 6 --n_bootstrap 500000 \
  --output_csv bootstrap_p6.csv --output_md bootstrap_p6.md
```

`compute_sample_errors_v2.py` stores a `trip_ids` array alongside the errors;
`make_paired_bootstrap_v2.py` resamples over it. Without that array it falls
back to resampling individual windows and warns, because treating correlated
windows from one trip as independent understates the variance.

## Relationship to the numbers in the paper

Measured reproduction of the paper's Porto results with this code, three seeds
(2024/2025/2026), the deployed configuration, and `run_porto_main.sh`:

| Quantity | This repository | Paper |
|---|---|---|
| Parameters, `H=6` / `H=12` | 11,129 / 11,993 | 11,129 / 11,993 |
| Test ADE, `H=6` | 234.03 +/- 0.28 | 234.08 +/- 0.18 |
| Test ADE, `H=12` | 439.07 +/- 1.79 | 432.97 +/- 0.43 |
| Matched-subset trips, `H=6` / `H=12` | 119,110 / 114,208 | 119,110 / 114,208 |

`verify_model_config.py` checks the parameter budget component by component
(1,120 backbone + 2,247 gate + 7,762 residual head at `H=6`) and the closed-form
prior algebra, including the component-wise clipping in P6.

Two caveats before you compare a run against the paper:

* **`H=12` lands about 1.4% above the published ADE.** The deterministic
  baselines rebuilt from the public Porto dump sit 0.07-0.25% above their
  published values, which accounts for well under a metre of the gap. We did not
  isolate the remainder. `H=6` is within 0.02%.
* **The Porto dump matters.** Rebuilding from the public source reproduces the
  trip-level split and the 131,072-window matched subset exactly, but roughly
  20% of individual windows differ from the arrays behind the published tables,
  so per-window numbers will not match bit for bit.

## Notes

* `analysis/make_paired_bootstrap.py` resamples whole trips and reports, for
  every comparison, the observed tail count and a 95% bootstrap interval on the
  ADE difference at a configurable `--n_bootstrap`.
* Trained checkpoints are not included; every result has to be produced by
  running the commands above.
* Device: `primotraj/main_traj.py` selects `cuda:0` automatically whenever
  CUDA is available and falls back to CPU otherwise; the baselines take an
  explicit `--use_gpu`. `USE_GPU=0 bash run_porto_main.sh` forces every entry
  point onto the CPU. The code is device-agnostic, but this package was
  smoke-tested on CPU only.

## License

MIT, see `LICENSE`.
