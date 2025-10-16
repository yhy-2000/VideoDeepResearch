

























export API_MODEL_NAME=deepseek-r1-250120
export API_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
export API_KEY=3dbe8ff1-579f-46e1-ba2d-e1fa1d0f971c

export API_MODEL_NAME_VLM=./pretrained_model/Qwen2.5-VL-7B-Instruct
export API_BASE_URL_VLM=http://localhost:12345/v1
export API_KEY_VLM=EMPTY

python eval_qwen25vl.py --dataset LongVideoBench --dataset_mode '_val' --dataset_folder ./long_video/benchmark/formulated




