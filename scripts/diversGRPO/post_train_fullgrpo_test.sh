#!/bin/bash

MAMBA_ENV="tina"
# eval "$(mamba shell hook --shell bash)" && mamba activate "${MAMBA_ENV}"
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

source "./scripts/set/set_vars.sh"

export CUDA_VISIBLE_DEVICES=0,1 # Set the GPUs you want to use
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "Number of GPUs: ${GPU_COUNT}"
echo ""

MODEL_NAME="qwen_math_1.5b"
DATASET_NAME="salt_exp1" # baseline

## Main datasets
#DATASET_NAME="curated_deepscaler"
#DATASET_NAME="curated_still"
#DATASET_NAME="curated_open_rs3"
#DATASET_NAME="curated_open_rs2"
#DATASET_NAME="curated_open_rs1"

## Extra datasets
#DATASET_NAME="curated_limr"
#DATASET_NAME="curated_open_r1"
#DATASET_NAME="curated_thoughts"

## Ablation
#DATASET_NAME="curated_limr_large_lr_ablation"
#DATASET_NAME="curated_limr_small_lr_ablation"
#DATASET_NAME="curated_limr_large_rank_ablation"
# DATASET_NAME="curated_limr_medium_rank_ablation"
#DATASET_NAME="curated_limr_small_rank_ablation"
#DATASET_NAME="curated_limr_tiny_rank_ablation"
#DATASET_NAME="curated_open_rs3_drgrpo_ablation"

PY_SCRIPT="./tina/post_train_hf/diversGRPO.py"
PY_CONFIG="./recipes/${MODEL_NAME}/diversGRPO/model_${DATASET_NAME}.yaml"
ACCELERATE_DS_CONFIG="./recipes/accelerate_ds_cfgs/ds_zero2.yaml"

echo ""
echo "Running ${PY_SCRIPT} on model ${MODEL_NAME} with dataset ${DATASET_NAME}"
echo ""

# Function to check if a port is in use
is_port_in_use() {
    netstat -tuln | grep -q ":$1 "
    return $?
}

# Find available port starting from 29510
PORT=29510
while is_port_in_use $PORT; do
    echo "Port $PORT is in use, trying the next port"
    PORT=$((PORT + 1))
done
echo "Using port: $PORT"

if [[ "${DATASET_NAME}" == "curated_thoughts" || "${DATASET_NAME}" == "curated_open_r1" || "${DATASET_NAME}" == "curated_open_rs3" || "${DATASET_NAME}" == "curated_open_rs3_drgrpo_ablation" ]]; then
        ACCELERATE_LOG_LEVEL=info accelerate launch \
                --config_file "${ACCELERATE_DS_CONFIG}" \
                --main_process_port=$PORT \
                --num_processes="${GPU_COUNT}" "${PY_SCRIPT}" --config "${PY_CONFIG}" --cosine_max_len 3584
else
        ACCELERATE_LOG_LEVEL=info accelerate launch \
                --config_file "${ACCELERATE_DS_CONFIG}" \
                --main_process_port=$PORT \
                --num_processes="${GPU_COUNT}" "${PY_SCRIPT}" --config "${PY_CONFIG}" --cosine_max_len 4096
fi

echo "END TIME: $(date)"
echo "DONE"
