cd eval/preprocess

export HF_ENDPOINT=https://hf-mirror.com
DATASET_DIR=./benchmark

# Step 1: extracting video frames at N fps (default N=2)
# Optional: frames will be extracted automatically if not preprocessed
echo 'extracting frames....'
python extract_frames.py --dataset mlvu

# Step 2: cutting videos into clips of M seconds (default M=10)
echo 'cutting video clips....'
total_threads=96
for i in $(seq $(($1*32)) $(($1*32+31))); do
    mkdir -p ./tmp_clips/$i
    cd ./tmp_clips/$i
    python cut_videos_moviepy.py --thread_num $total_threads --thread_idx $i --clip_duration 30 --dataset lvbench --dataset_mode '' --dataset_folder $DATASET_DIR&
done
wait
echo 'cut video clips done!'

# Step 3: calculating video clip embeddings with languagebind
echo 'calculating video clip embeddings....'
for i in $(seq 0 31); do
    CUDA_VISIBLE_DEVICES=$((i%8)) python /share/project/huaying/VideoDeepResearch/eval/calculate_video_clip_embedding.py --thread_num 32 --thread_idx $i --dataset MLVU --clip_duration 10 --dataset_folder $DATASET_DIR&
done
wait


