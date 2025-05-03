#!/bin/bash
set -x


# 0. Default parameters
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"CoT"}  # Experiment name, e.g., "CoT", "WHWM"
DATASET_NAME=${DATASET_NAME:-"gsm8k"}  # Dataset name, e.g., "gsm8k", "aqua", "math"
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-7B-Instruct"}
SAVE_CHECKPOINT_PATH=${SAVE_CHECKPOINT_PATH:-"/models/shanwen/${EXPERIMENT_NAME}_${DATASET_NAME}_$(basename ${MODEL_PATH})"}


# 1. Prepare the dataset
export HF_ENDPOINT="https://hf-mirror.com"

train_file="./data/${DATASET_NAME}/train.parquet"
val_file="./data/${DATASET_NAME}/test.parquet"
python3 -m src.data_preprocess.${DATASET_NAME}.${EXPERIMENT_NAME} --local_dir ./data/${DATASET_NAME}


# 2. Perform the training
export VERL_USE_MODELSCOPE=True
python3 -m src.train \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=$train_file \
    data.val_files=$val_file \
    data.train_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    custom_reward_function.path=./src/reward_score/${DATASET_NAME}.py \
    custom_reward_function.name=compute_${EXPERIMENT_NAME}_score \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    trainer.logger=['console'] \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=30 \
    trainer.test_freq=30 \
    trainer.default_local_dir=${SAVE_CHECKPOINT_PATH} \
    trainer.total_epochs=3 $@