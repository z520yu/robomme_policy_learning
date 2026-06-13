#!/usr/bin/env bash
set -u

RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
BATCHES="${BATCHES:-8 16 32 64}"
STEPS="${STEPS:-5}"
GPUS="${GPUS:-0,1,2,3}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOGROOT="${LOGROOT:-runs/train_ramp/logs/${RUN_ID}}"
CKPT_BASE="${CKPT_BASE:-/home/zhangyu/robomme_train_ramp_no_ckpt}"

mkdir -p "${LOGROOT}" runs/train_ramp
echo "${RUN_ID}" > runs/train_ramp/latest_run_id.txt
echo "${LOGROOT}" > runs/train_ramp/latest_logroot.txt

echo "RUN_ID=${RUN_ID}"
echo "BATCHES=${BATCHES}"
echo "STEPS=${STEPS}"
echo "GPUS=${GPUS}"
echo "NUM_WORKERS=${NUM_WORKERS}"
echo "LOGROOT=${LOGROOT}"

for BATCH in ${BATCHES}; do
  EXP="framesamp_modul_ramp_b${BATCH}_${RUN_ID}"
  LOG="${LOGROOT}/batch_${BATCH}.log"
  MON="${LOGROOT}/batch_${BATCH}_gpu.csv"
  STATUS_FILE="${LOGROOT}/batch_${BATCH}.status"

  echo "START batch=${BATCH} exp=${EXP}"
  echo "timestamp,index,memory.used.MiB,utilization.gpu.percent" > "${MON}"

  env \
    CUDA_VISIBLE_DEVICES="${GPUS}" \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    "${ROBOMME_POLICY_VENV}/bin/python" scripts/run_train_no_ckpt.py mme_vla_suite \
      --exp-name "${EXP}" \
      --overwrite \
      --seed 7 \
      --batch-size "${BATCH}" \
      --num-workers "${NUM_WORKERS}" \
      --fsdp-devices 4 \
      --num-train-steps "${STEPS}" \
      --log-interval 1 \
      --save-interval 999999 \
      --keep-period 999999 \
      --dataset-path data/robomme_preprocessed_data_sample \
      --checkpoint-base-dir "${CKPT_BASE}" \
      --model.use-history \
      --model.history-config perceptual-framesamp-modul.yaml \
      --no-wandb-enabled > "${LOG}" 2>&1 &
  TRAIN_PID=$!

  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    TS="$(date '+%F %T')"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
      while IFS= read -r line; do
        case "${line}" in
          0,*|1,*|2,*|3,*) echo "${TS},${line}" >> "${MON}" ;;
        esac
      done
    sleep 2
  done

  wait "${TRAIN_PID}"
  STATUS=$?
  echo "${STATUS}" > "${STATUS_FILE}"
  echo "DONE batch=${BATCH} status=${STATUS}"

  if [ "${STATUS}" -ne 0 ]; then
    echo "STOP after failed batch=${BATCH}"
    break
  fi
done
