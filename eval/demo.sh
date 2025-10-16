#!/bin/bash
cd ./eval


# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com
DATASET_DIR=/share/project/huaying/benchmark

export API_MODEL_NAME_TEMPORAL_GROUNDING=/share/project/huaying/VideoDeepResearch/train/LLaMA-Factory-main/saves/qwen25-7b-sft-temporal_grounding_real_6k_base
export API_BASE_URL_TEMPORAL_GROUNDING=http://localhost:22345/v1
export API_KEY_TEMPORAL_GROUNDING=EMPTY

# export API_MODEL_NAME=/share/project/huaying/VideoDeepResearch/train/LLaMA-Factory-dpo/saves/qwen25-3b-dpo_1k/checkpoint-40/full_model_3b
export API_MODEL_NAME=/share/project/huaying/VideoDeepResearch/train/LLaMA-Factory-dpo/saves/qwen25-7b-dpo_1k/checkpoint-40/full_model
export API_BASE_URL=http://localhost:22346/v1
export API_KEY=EMPTY

export API_MODEL_NAME_VLM=/share/project/minghao/Models/Qwen2.5-VL-7B-Instruct


# 启动第一个服务（planner）
CUDA_VISIBLE_DEVICES=0 python servers_io/vllm_server_planner.py > ./vllm_log/planner.log 2>&1 &
planner_pid=$!

# 启动第二个服务（temporal grounder）
CUDA_VISIBLE_DEVICES=1 python servers_io/vllm_server_temporal_grounder.py > ./vllm_log/temporal_grounder.log 2>&1 &
grounder_pid=$!

# 等待服务启动
echo "Waiting for servers to start..."
sleep 10




bash ./clear.sh
export TOPK=10
threads=6
mkdir -p ./logs


# 运行demo脚本
CUDA_VISIBLE_DEVICES=2 python demo.py \
    --video_path /share/project/huaying/benchmark/lvbench/videos/_T2Avd3tFHc.mp4 \
    --question "What is the main topic of this video?" \
    --options "A. Technology" "B. Nature" "C. Sports" "D. Politics" \
    --topk $TOPK | tee -a ./logs/demo.log

# # 运行评估脚本
# for dataset in mlvu; do
#     for i in $(seq 0 $(($threads-1))); do
#         log_file=./logs/eval_${dataset}_thread${i}.log
#         CUDA_VISIBLE_DEVICES=$((i+2)) python eval.py \
#             --dataset $dataset \
#             --dataset_folder $DATASET_DIR \
#             --thread_num $threads \
#             --thread_idx $i \
#             --clip_duration 10 \
#             --end_sample_number 100000 | tee -a $log_file &
#     done
# done

# wait
# echo "✅ All evaluation threads finished."



pkill -f vllm_server_planner.py
pkill -f vllm_server_temporal_grounder.py