import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from retriever_languagebind import Retrieval_Manager
from languagebind import LanguageBind, to_device, transform_dict, LanguageBindImageTokenizer, LanguageBindVideoTokenizer
import torch
import numpy as np
import cv2

import pickle
import time
import json
from moviepy.editor import VideoFileClip, concatenate_videoclips
from decord import VideoReader, cpu
from tqdm import tqdm

import math
import argparse
import subprocess
import datetime
import multiprocessing
import os

            
def is_valid_frame(frame_path):
    if not os.path.exists(frame_path):
        return False
    try:
        with Image.open(frame_path) as img:
            img.verify() 
        with Image.open(frame_path) as img:
            img.load()  
        return True
    except:
        return False



def extract_video_clip(video_path: str, start: float, end: float) -> str:
    with VideoFileClip(video_path) as clip:
        duration = clip.duration
        start = max(0, min(start, duration))
        end = max(0, min(end, duration))
        if end <= start:
            start = end-1

        subclip = clip.subclip(start, end)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)
        subclip.write_videofile(tmp_path, codec="libx264", audio_codec="aac", logger=None)
    return tmp_path


def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def timestamp_to_clip_path(begin_time_stamp, end_time_stamp, video_path, fps=2):
    video_id = video_path.split('/')[-1][:-4]
    frame_folder = os.path.join(args.dataset_folder, f'dense_frames/{video_id}/')

    if (end_time_stamp - begin_time_stamp) < 1:
        begin_time_stamp = max(begin_time_stamp - 0.5, 0)
        end_time_stamp = end_time_stamp + 0.5

    num_frames = int((end_time_stamp - begin_time_stamp) * fps)
    time_points = [begin_time_stamp + i * (1.0 / fps) for i in range(num_frames)]

    frames = []
    for time_point in time_points:
        candidates = [file for file in os.listdir(frame_folder) if file.startswith("frame_") and (file.endswith(".png") or file.endswith(".jpg"))]
        if not candidates:
            continue

        closest_frame = min(
            candidates,
            key=lambda file: abs(float(file.replace("frame_", "").replace(".png", "").replace(".jpg", "")) - time_point)
        )

        frame_path = os.path.join(frame_folder, closest_frame)
        try:
            Image.open(frame_path)
            frames.append(frame_path)
        except:
            pass

    if len(frames) > MAX_FRAMES:
        chunk_size = len(frames) // MAX_FRAMES
        frames = frames[::chunk_size]
    return frames[:MAX_FRAMES]




def clip_number_to_clip_path(clip_numbers, video_path, clip_duration=10, fps=2):
    video_id = os.path.basename(video_path)[:-4]
    frame_folder = os.path.join(args.dataset_folder, 'dense_frames', video_id)

    frames = []
    for clip_number in clip_numbers:
        begin_time_stamp = clip_number * clip_duration
        end_time_stamp = begin_time_stamp + clip_duration

        time_points = [begin_time_stamp + i * (1.0 / fps) for i in range(int(clip_duration * fps))]

        for time_point in time_points:
            try:
                frame_name = sorted(
                    [file for file in os.listdir(frame_folder) if file.endswith(('.jpg', '.png'))],
                    key=lambda file: abs(float(file.replace('.png','').replace('.jpg','').split('_')[1]) - time_point)
                )[0]
                frame_path = os.path.join(frame_folder, frame_name)
                Image.open(frame_path)
                frames.append(frame_path)
            except Exception as e:
                continue

    if len(frames) > MAX_FRAMES:
        interval = len(frames) / MAX_FRAMES
        frames = [frames[int(i * interval)] for i in range(MAX_FRAMES)]

    if len(frames) < MIN_FRAMES and len(frames) > 0:
        interval = len(frames) / MIN_FRAMES
        frames = [frames[int(i * interval)] for i in range(MIN_FRAMES)]

    return frames


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, default='./benchmark')
    parser.add_argument('--dataset', type=str, default='mlvu')
    parser.add_argument('--dataset_mode', type=str, default='')
    parser.add_argument('--retriever_type', type=str, default='large')
    parser.add_argument('--clip_duration', type=int, default=10)
    parser.add_argument('--thread_num', type=int, default=1)
    parser.add_argument('--thread_idx', type=int, default=0)
    args = parser.parse_args()

    args.dataset_folder = os.path.join(args.dataset_folder, args.dataset)
    retriever = Retrieval_Manager(args=args, clip_save_folder=f'{args.dataset_folder}/clips/{args.clip_duration}',clip_duration=args.clip_duration)
    retriever.load_model_to_gpu()
    data_li = json.load(open(f'{args.dataset_folder}/qa{args.dataset_mode}.json'))

    video_li = sorted(list(set([dic['video_path']+'$'+str(dic['duration']) for dic in data_li])))

    print(f'{args.dataset_folder}/qa{args.dataset_mode}.json', len(data_li), len(video_li))

    chunk_size = len(video_li)//args.thread_num + 1


    for i,video_path in enumerate(tqdm(video_li[chunk_size*args.thread_idx:chunk_size *(args.thread_idx+1)])):
        video_path,duration = video_path.split('$')
        video_name = video_path.split('/')[-1].split('.')[0]
        folder_path = f'{args.dataset_folder}/embeddings/{args.clip_duration}/{args.retriever_type}'
        os.makedirs(folder_path, exist_ok=True)
        retriever.calculate_video_clip_embedding(video_path, folder_path, total_duration=eval(duration), pre_calculate=True)

