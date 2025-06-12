source activate /share/huaying/envs/llama-factory
cd eval

# 多进程切分视频
echo 'cutting video clips....'
for i in $(seq 0 15); do
    python cut_videos_moviepy.py --thread_num 16  --thread_idx $i --clip_duration 10 --dataset LongVideoBench --dataset_mode '_val' --dataset_folder DATASET_DIR&
done
wait
echo 'cut video clips done!'


# 多进程计算视频片段embedding
echo 'calculating video clip embedding....'
for i in $(seq 0 15); do
    CUDA_VISIBLE_DEVICES=$((i%8))  python calculate_video_clip_embedding.py --thread_num 16 --thread_idx $i --dataset LongVideoBench --dataset_mode '_val' --clip_duration 10 --dataset_folder DATASET_DIR&
done
wait


# 运行测试脚本
python prompt_seed15vl.py --dataset LongVideoBench --dataset_mode '_val' --dataset_folder DATASET_DIR