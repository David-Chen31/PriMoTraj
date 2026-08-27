#!/usr/bin/env bash
set -euo pipefail
# Matched-capacity mixture-of-experts control.
#
# Identical encoder, identical sample-wise router, identical anchoring at p_T and
# identical total parameter budget as the deployed PriMoTraj; the only change is
# that the seven experts are learned functions of the input window instead of the
# closed-form motion priors P1-P7. The bounded residual head is removed and its
# parameter budget is spent on the experts, which is what keeps the two models
# the same size (11,171 vs 11,129 at H=6; 11,923 vs 11,993 at H=12).
R="${ROOT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PY="${PY:-python}"
LR="${LR:-5e-5}"; EPOCHS="${EPOCHS:-5}"; BATCH="${BATCH:-2048}"
SEEDS="${SEEDS:-2024 2025 2026}"; PRED_LENS="${PRED_LENS:-6 12}"
VARIANT="${VARIANT:-learned}"          # learned | closed_form
LOG_DIR="${LOG_DIR:-$R/logs/moe_control}"
CKPT="${CKPT:-$R/baselines/checkpoints/moe_control}"
mkdir -p "$LOG_DIR" "$CKPT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-6}"

for seed in $SEEDS; do for H in $PRED_LENS; do
  if [ "$H" = "6" ]; then EH=40; else EH=30; fi
  if [ "$VARIANT" = "learned" ]; then RHEAD=none; else RHEAD=full_window; fi
  tag="${VARIANT}_seed${seed}_pred${H}"
  [ -s "$LOG_DIR/$tag.log" ] && grep -q '^BENCHMARK|' "$LOG_DIR/$tag.log" && { echo "[SKIP] $tag"; continue; }
  echo "[RUN] $tag  lr=$LR epochs=$EPOCHS expert_hidden=$EH residual=$RHEAD"
  "$PY" -u ""$R/primotraj/main_traj.py"" \
    --seed "$seed" --data ""$R/data/porto_ds15.npz"" \
    --checkpoint_dir "$CKPT" --name "$tag" \
    --seq_len 12 --pred_len "$H" --train_epochs "$EPOCHS" --batch_size "$BATCH" \
    --learning_rate "$LR" \
    --use_mdm True --use_moe False --motion_prior cvmix --motion_prior_weight 1.0 \
    --motion_prior_recent_weight 0.5 --motion_prior_mode gate --motion_prior_damping 0.8 \
    --motion_prior_n_priors 7 --motion_prior_gate_hidden 64 --motion_prior_gate_horizon 3 \
    --motion_prior_experts "$VARIANT" --motion_prior_expert_hidden "$EH" \
    --motion_prior_residual_head "$RHEAD" --motion_prior_residual_hidden 64 \
    --motion_prior_residual_init 0.15 \
    --mse_loss_weight 0.1 --fde_loss_weight 1.0 --ade_loss_weight 0.0 \
    --weight_decay 1e-4 --min_learning_rate 1e-6 --grad_clip_norm 1.0 --progress False \
    > "$LOG_DIR/$tag.log" 2>&1
  grep -E '^BENCHMARK\|' "$LOG_DIR/$tag.log" | head -1 | cut -c1-120
done; done
