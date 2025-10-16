cd train/LLaMA-Factory-main
WANDB_MODE=disabled MASTER_PORT=29500 FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/qwen_planner.yaml
