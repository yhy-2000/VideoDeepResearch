cd train/LLaMA-Factory-main
export EXPERIMENT_NAME=qwen25-7b-train-dpo
WANDB_MODE=disabled MASTER_PORT=29500 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen_planner_dpo_3b.yaml \
    2>&1 | tee log_files/${EXPERIMENT_NAME}_${date_time}.log


