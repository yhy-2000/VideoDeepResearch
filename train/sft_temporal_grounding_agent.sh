cd train/LLaMA-Factory-main
WANDB_MODE=disabled MASTER_PORT=29500 llamafactory-cli train examples/train_full/qwen_temporal_grounding.yaml
