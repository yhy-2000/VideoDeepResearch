VLLM_USE_MODELSCOPE=false CUDA_VISIBLE_DEVICES=0 vllm serve /share/huaying/pretrained_model/Qwen2.5-VL-7B-Instruct \
    --port 12345 \
    --max-model-len 100000 \
    --tensor-parallel-size 1 \
    --limit-mm-per-prompt 'image=1000,video=100' \
    --enable-chunked-prefill True \
    --gpu-memory-utilization 0.7

    