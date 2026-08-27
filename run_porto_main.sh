#!/usr/bin/env bash
set -euo pipefail

# Porto main comparison: PriMoTraj plus every baseline whose code ships in this
# package, over 3 seeds and both horizons (T=12, H in {6,12}).
#
#   DATA=data/porto_ds15.npz bash run_porto_main.sh
#
# Then aggregate:
#   python collect_results.py logs --csv results.csv

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
DATA="${DATA:-${ROOT_DIR}/data/porto_ds15.npz}"
SEEDS="${SEEDS:-2024 2025 2026}"
PRED_LENS="${PRED_LENS:-6 12}"
SEQ_LEN="${SEQ_LEN:-12}"
EPOCHS="${EPOCHS:-5}"
BATCH="${BATCH:-2048}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
CKPT_DIR="${CKPT_DIR:-${ROOT_DIR}/checkpoints}"
USE_GPU="${USE_GPU:-1}"
MODELS="${MODELS:-primotraj lstm dlinear last cv cvmix cv_kf}"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}"
gpu_flag=""
if [[ "${USE_GPU}" == "1" ]]; then
  gpu_flag="--use_gpu"
else
  # The baselines take an explicit --use_gpu flag, but primotraj/main_traj.py
  # selects cuda automatically whenever it is available; hiding the devices is
  # what actually forces every entry point onto the CPU.
  export CUDA_VISIBLE_DEVICES=""
fi

if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then export OMP_NUM_THREADS=8; fi
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"

echo "DATA=${DATA}"
echo "MODELS=${MODELS}"
echo "SEEDS=${SEEDS}  PRED_LENS=${PRED_LENS}  SEQ_LEN=${SEQ_LEN}  EPOCHS=${EPOCHS}  BATCH=${BATCH}"
echo

for pred_len in ${PRED_LENS}; do
  for model in ${MODELS}; do
    case "${model}" in
      last|cv|cvmix|cv_kf)
        # Deterministic baselines: no training, no seed dependence.
        log="${LOG_DIR}/Physics_${model}_pred${pred_len}.log"
        echo "[RUN] ${model} H=${pred_len} -> ${log}"
        tune=""
        if [[ "${model}" == "cv_kf" ]]; then tune="--tune_kf --tune_max_batches 160"; fi
        ${PYTHON} -u "${ROOT_DIR}/baselines/physics_baselines.py" \
          --data_path "${DATA}" --seq_len "${SEQ_LEN}" --pred_len "${pred_len}" \
          --batch_size "${BATCH}" --mode "${model}" \
          --recent_steps 3 --mix_weight 0.5 ${tune} ${gpu_flag} \
          2>&1 | tee "${log}"
        continue
        ;;
    esac

    for seed in ${SEEDS}; do
      log="${LOG_DIR}/${model}_seed${seed}_pred${pred_len}.log"
      echo "[RUN] ${model} seed=${seed} H=${pred_len} -> ${log}"
      case "${model}" in
        primotraj)
          # Deployed configuration: 7 priors, h_g=64, K_g=3,
          # full-window residual head with a learned per-step bound.
          ${PYTHON} -u "${ROOT_DIR}/primotraj/main_traj.py" \
            --seed "${seed}" --data "${DATA}" \
            --checkpoint_dir "${CKPT_DIR}/PriMoTraj" --name "seed${seed}_pred${pred_len}" \
            --seq_len "${SEQ_LEN}" --pred_len "${pred_len}" \
            --train_epochs "${EPOCHS}" --batch_size "${BATCH}" --learning_rate 5e-5 \
            --use_mdm True --use_moe False \
            --motion_prior cvmix --motion_prior_weight 1.0 \
            --motion_prior_mode gate --motion_prior_recent_weight 0.5 \
            --motion_prior_damping 0.8 \
            --motion_prior_n_priors 7 \
            --motion_prior_gate_hidden 64 --motion_prior_gate_horizon 3 \
            --motion_prior_residual_head full_window \
            --motion_prior_residual_hidden 64 --motion_prior_residual_init 0.15 \
            --mse_loss_weight 0.1 --fde_loss_weight 1.0 --ade_loss_weight 0.0 \
            --weight_decay 1e-4 --grad_clip_norm 1.0 --min_learning_rate 1e-6 \
            --progress False \
            2>&1 | tee "${log}"
          ;;
        lstm)
          ${PYTHON} -u "${ROOT_DIR}/GRU/main_traj.py" \
            --data_path "${DATA}" --seq_len "${SEQ_LEN}" --pred_len "${pred_len}" \
            --batch_size "${BATCH}" --epochs "${EPOCHS}" --lr 1e-3 --seed "${seed}" \
            --rnn_type LSTM --hidden_size 64 --num_layers 1 --dropout 0.0 ${gpu_flag} \
            --save_path "${CKPT_DIR}/LSTM/seed${seed}_pred${pred_len}.pt" \
            2>&1 | tee "${log}"
          ;;
        dlinear)
          ${PYTHON} -u "${ROOT_DIR}/DLinear/main_traj.py" \
            --data_path "${DATA}" --seq_len "${SEQ_LEN}" --pred_len "${pred_len}" \
            --batch_size "${BATCH}" --epochs "${EPOCHS}" --seed "${seed}" ${gpu_flag} \
            2>&1 | tee "${log}"
          ;;
        *)
          echo "unknown model: ${model}" >&2; exit 1 ;;
      esac
    done
  done
done

echo
echo "All runs finished. Aggregate with:"
echo "  ${PYTHON} ${ROOT_DIR}/collect_results.py ${LOG_DIR} --csv ${ROOT_DIR}/results.csv"
