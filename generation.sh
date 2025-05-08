#!/bin/bash
set -x


# 0. Default parameters
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"CoT"}  # Experiment name, e.g., "CoT", "WHWM"
DATASET_NAME=${DATASET_NAME:-"gsm8k"}  # Dataset name, e.g., "gsm8k", "aqua", "math"
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}
FSDP_CHECKPOINT_PATH=${FSDP_CHECKPOINT_PATH:-""}  # Optional FSDP checkpoint path
OUTPUT_DIR=${OUTPUT_PATH:-"/datasets/shanwen/${EXPERIMENT_NAME}/${DATASET_NAME}/$(basename ${MODEL_PATH})/"}
OUTPUT_FILENAME=${OUTPUT_FILENAME:-"test.parquet"}
OUTPUT_PATH=${OUTPUT_PATH:-"${OUTPUT_DIR}/${OUTPUT_FILENAME}"}


# 1. Prepare the dataset
export HF_ENDPOINT="https://hf-mirror.com"

val_file="./data/${DATASET_NAME}/test.parquet"
python3 -m src.data_preprocess.${DATASET_NAME}.${EXPERIMENT_NAME} --local_dir ./data/${DATASET_NAME}


# 2. Convert the FSDP checkpoint to HF checkpoint
if [ -n "$FSDP_CHECKPOINT_PATH" ]; then
    converted_model_path="$(dirname ${FSDP_CHECKPOINT_PATH})/hf_model"
    if [[ -d "$converted_model_path" ]]; then
        echo "Converted model already exists at $converted_model_path"
    else
        echo "Converting FSDP checkpoint to HF checkpoint..."
        python3 -m src.convert_fsdp_to_hf \
            --fsdp_checkpoint_path $FSDP_CHECKPOINT_PATH \
            --hf_model_path $MODEL_PATH \
            --output_path $converted_model_path
        echo "Converted model saved at $converted_model_path"
    fi
    MODEL_PATH=$converted_model_path
fi


# 3. Perform the generation
export VERL_USE_MODELSCOPE=True
python3 -m src.generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$val_file \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$OUTPUT_PATH \
    model.path=$MODEL_PATH \
    +model.trust_remote_code=True \
    rollout.temperature=0.7 \
    rollout.top_k=20 \
    rollout.top_p=0.9 \
    rollout.prompt_length=4096 \
    rollout.response_length=4096 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.8 $@


# 4. Evaluate the generation
python3 -m src.eval.${DATASET_NAME} --data_path $OUTPUT_PATH