import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
import random
import math
import os
from decord import VideoReader, cpu
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import pandas as pd
from huggingface_hub import snapshot_download
import json
import cv2
# 设置 Hugging Face 镜像地址
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def format_time(seconds):
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    return f"{int(hours):02d}-{int(mins):02d}-{int(secs):02d}"


def check_video_openable(video_path):
    # 尝试打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        # print("无法打开视频文件")
        return False
    
    # 获取视频的帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 如果帧数大于0，认为视频文件有效
    if total_frames > 0:
        # print("视频文件可以成功打开并包含帧")
        cap.release()
        return True
    else:
        # print("视频文件没有帧")
        cap.release()
        return False



def custom_ffmpeg_extract_subclip(filename, t1, t2, targetname):
    flag, retry = 0, 0

    while flag != 1 and retry < 5:
        folder = os.path.dirname(targetname)
        os.makedirs(folder, exist_ok=True)

        try:
            # Extract subclip
            video = VideoFileClip(filename)
            video_sub_clip = video.subclip(t1, t2)

            # Resize video if necessary
            width, height = video.size
            scale = min(700 / max(width, height), 1)  # Ensure resizing only when needed
            if scale < 1:
                video_sub_clip = video_sub_clip.resize(scale)

            video_sub_clip = concatenate_videoclips([video_sub_clip])
            video_sub_clip.write_videofile(targetname, codec="libx264", audio_codec="aac", logger=None)

            # Validate with decord
            VideoReader(targetname, ctx=cpu(0), num_threads=1)
            flag = 1
        except Exception as e:
            print(f"Attempt {retry + 1} failed: {e}")
            retry += 1

    if not flag:
        print(f"Failed to process {targetname} after {retry} retries.")


def process_clip(arg):
    start_time, end_time, video_path, clip_save_path = arg
    custom_ffmpeg_extract_subclip(
        video_path,
        start_time,
        end_time,
        clip_save_path
    )


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mlvu')
    parser.add_argument('--dataset_mode', type=str, default='')
    parser.add_argument('--dataset_folder', type=str, default='')
    parser.add_argument('--clip_duration', type=int, default=10)
    parser.add_argument('--thread_num', type=int, default=1)
    parser.add_argument('--thread_idx', type=int, default=0)
    args = parser.parse_args()


    folder = f'{args.dataset_folder}/{args.dataset}/videos/'
    clip_save_path = f'{args.dataset_folder}/{args.dataset}/clips/{args.clip_duration}'
    
    video_li = [folder+file for file in os.listdir(folder)]
    print(len(video_li))
    chunk_size = len(video_li)//args.thread_num + 1

    invalid_num = 0
    for video_path in tqdm(video_li[chunk_size*args.thread_idx:chunk_size *(args.thread_idx+1)]):
        try:
            duration = VideoFileClip(video_path).duration
            chunk_number = math.ceil(duration/args.clip_duration)
            args_list = []
            video_name = video_path.split('/')[-1].split('.')[0]

            clip_paths = []
            for i in range(chunk_number):
                start_time, end_time = i*args.clip_duration, min(i*args.clip_duration+args.clip_duration, duration)
                start_time_s, end_time_s = format_time(start_time), format_time(end_time)
                clip = f'{clip_save_path}/{video_name}/clip_{i}_{start_time_s}_to_{end_time_s}.mp4'
                clip_paths.append(clip)
                if not os.path.exists(clip) or not check_video_openable(clip):
                    args_list.append((start_time, end_time, video_path, clip))

            # remove invalid clips
            os.makedirs(f'{clip_save_path}/{video_name}',exist_ok=True)
            for file in os.listdir(f'{clip_save_path}/{video_name}'):
                if f'{clip_save_path}/{video_name}/{file}' not in clip_paths:
                    os.remove(f'{clip_save_path}/{video_name}/{file}')

            if len(args_list)>1:  
                print(video_name,'videos to extract:',len(args_list))
                for args_ in tqdm(args_list,desc='cutting videos'):
                    process_clip(args_)
        except Exception as e:
            print(e)
    print(invalid_num)     




