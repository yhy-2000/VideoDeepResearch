import re
import json
import subprocess
from tqdm import tqdm
import os
import random
import time
import multiprocessing
import cv2
import decord
from PIL import Image

import os
import cv2
from PIL import Image

MAX_FRAMES = 32
MIN_FRAMES = 8  


def timestamp_to_clip_path(dataset_folder, begin_time_stamp, end_time_stamp, video_path, fps=2):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    frame_folder = os.path.join(dataset_folder, f'dense_frames/{video_id}/')
    os.makedirs(frame_folder, exist_ok=True)

    # 自动扩展时间范围，防止间隔太短
    if (end_time_stamp - begin_time_stamp) < 1:
        begin_time_stamp = max(begin_time_stamp - 0.5, 0)
        end_time_stamp += 0.5

    num_frames = int((end_time_stamp - begin_time_stamp) * fps)
    time_points = [begin_time_stamp + i * (1.0 / fps) for i in range(num_frames)]

    # 提取指定时间戳的帧
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    for t in time_points:
        frame_idx = int(round(t * video_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if success:
            frame_name = f"frame_{t:.3f}.jpg"
            frame_path = os.path.join(frame_folder, frame_name)
            if not os.path.exists(frame_path):  # 避免重复写入
                cv2.imwrite(frame_path, frame)
    cap.release()


    # 获取候选帧文件名
    candidates = [file for file in os.listdir(frame_folder)
                  if file.startswith("frame_") and (file.endswith(".png") or file.endswith(".jpg"))]
    if not candidates:
        return [], []

    frame_paths = []
    timestamps = []
    for time_point in time_points:
        closest_frame = min(
            candidates,
            key=lambda file: abs(float(file.replace("frame_", "").replace(".png", "").replace(".jpg", "")) - time_point)
        )
        frame_path = os.path.join(frame_folder, closest_frame)
        try:
            Image.open(frame_path)  # 校验图片能否打开
            frame_paths.append(frame_path)
            timestamps.append(time_point)
        except:
            pass

    # 截断至 MAX_FRAMES
    if len(frame_paths) > MAX_FRAMES:
        chunk_size = len(frame_paths) // MAX_FRAMES
        frame_paths = frame_paths[::chunk_size][:MAX_FRAMES]
        timestamps = timestamps[::chunk_size][:MAX_FRAMES]

    return frame_paths, timestamps



def clip_number_to_clip_path(dataset_folder, clip_numbers, video_path, clip_duration=10, fps=2):
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    frame_folder = os.path.join(dataset_folder, 'dense_frames', video_id)
    os.makedirs(frame_folder, exist_ok=True)

    frame_list = []
    second_list = []

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    for clip_number in clip_numbers:
        begin_time_stamp = clip_number * clip_duration
        end_time_stamp = begin_time_stamp + clip_duration
        time_points = [begin_time_stamp + i * (1.0 / fps) for i in range(int(clip_duration * fps))]

        # 提取时间戳对应帧
        for t in time_points:
            frame_idx = int(round(t * video_fps))
            frame_name = f"frame_{t:.3f}.jpg"
            frame_path = os.path.join(frame_folder, frame_name)

            if not os.path.exists(frame_path):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = cap.read()
                if success:
                    cv2.imwrite(frame_path, frame)

        # 加载帧路径
        candidates = [file for file in os.listdir(frame_folder) if file.endswith(('.jpg', '.png'))]
        for time_point in time_points:
            try:
                frame_name = min(
                    candidates,
                    key=lambda file: abs(float(file.replace('.png','').replace('.jpg','').split('_')[1]) - time_point)
                )
                frame_path = os.path.join(frame_folder, frame_name)

                Image.open(frame_path)  # 校验图像
                frame_list.append(frame_path)
                second_list.append(time_point)
            except Exception as e:
                print(f"Error at time {time_point:.3f}: {e}")
                continue

    cap.release()

    # 采样
    if len(frame_list) > MAX_FRAMES:
        interval = len(frame_list) / MAX_FRAMES
        frame_list = [frame_list[int(i * interval)] for i in range(MAX_FRAMES)]
        second_list = [second_list[int(i * interval)] for i in range(MAX_FRAMES)]

    if len(frame_list) < MIN_FRAMES and len(frame_list) > 0:
        interval = len(frame_list) / MIN_FRAMES
        frame_list = [frame_list[int(i * interval)] for i in range(MIN_FRAMES)]
        second_list = [second_list[int(i * interval)] for i in range(MIN_FRAMES)]

    if len(frame_list) == 0:
        raise KeyError(f'Frame list {dataset_folder} {clip_numbers} {video_path} is invalid!')

    return frame_list, second_list




import av
import json
import math
from multiprocessing import Pool
from functools import partial
import torch
import numpy as np
import shutil

import shutil
import torch
import decord
from PIL import Image
from multiprocessing import Pool, cpu_count
import base64
import os
import cv2
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
from PIL import Image
import tempfile
from moviepy.video.io.VideoFileClip import VideoFileClip


MAX_DS_ROUND=20
MAX_FRAMES=32
MIN_FRAMES=10
MAX_IMAGE_RESOLUTION=748

def is_valid_video(path):
    try:
        cap = cv2.VideoCapture(path)
    except:
        return False

    if not cap.isOpened():
        return False

    try:
        video_reader = decord.VideoReader(path, num_threads=1)
        return True
    except:
        return False



def parse_subtitle_time(time_str):
    h, m, s_ms = time_str.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


# 提取给定视频在 start 到 end 秒之间的片段，保存到临时路径并返回路径。
def extract_video_clip(video_path: str, start: float, end: float) -> str:
    with VideoFileClip(video_path) as clip:
        duration = clip.duration
        # 校正时间范围
        start = max(0, min(start, duration))
        end = max(0, min(end, duration))
        if end <= start:
            start = end-1

        subclip = clip.subclip(start, end)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(tmp_fd)
        subclip.write_videofile(tmp_path, codec="libx264", audio_codec="aac", logger=None)
    return tmp_path


def load_subtitles(video_path):
    subtitle_path = video_path.replace('videos','subtitles').replace('.mp4','.srt')
    if os.path.exists(subtitle_path):
        subtitles = {}
        with open(subtitle_path, "r", encoding="utf-8") as file:
            content = file.read().split("\n\n")
            for section in content:
                if section.strip():
                    lines = section.split("\n")
                    if len(lines) >= 3:
                        time_range = lines[1].split(" --> ")
                        start_time = parse_subtitle_time(time_range[0])
                        end_time = parse_subtitle_time(time_range[1])
                        text = " ".join(line for line in lines[2:])
                        subtitles[(start_time, end_time)] = text
    else:
        subtitle_path = video_path.replace('videos','subtitles').replace('.mp4','_en.json')
        data_li = json.load(open(subtitle_path))
        subtitles = {}
        for dic in data_li:
            start_time = parse_subtitle_time(dic["start"])
            end_time = parse_subtitle_time(dic["end"])
            subtitles[(start_time, end_time)] = dic['line']
    
    return subtitles


def extract_subtitles(video_path):
    subtitles = load_subtitles(video_path)

    subtitle_frames = []
    for (start_time, end_time), text in subtitles.items():
        pattern = r'<font color="white" size=".72c">(.*?)</font>'
        raw_text = re.findall(pattern, text)
        try:
            text = raw_text[0]
            subtitle_frames.append((float(start_time), float(end_time), text))
        except:
            subtitle_frames.append((float(start_time), float(end_time), text))

    return subtitle_frames

def is_valid_frame(frame_path):
    if not os.path.exists(frame_path):
        return False
    try:
        with Image.open(frame_path) as img:
            img.verify()  # 检查文件完整性（但不解码完整图像数据）
        # verify 之后必须重新打开一次才能 load
        with Image.open(frame_path) as img:
            img.load()  # 强制解码整个图像数据，确保没有truncated
        return True
    except:
        return False



def load_image(image_path):
    # Read image
    frame = cv2.imread(image_path)
    try:
        # Resize image while maintaining aspect ratio
        height, width = frame.shape[:2]
        max_dimension = MAX_IMAGE_RESOLUTION
        
        # Calculate new dimensions
        if height > width:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        else:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
            
        # Resize the image
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Encode as JPEG and convert to base64
        _, buffer = cv2.imencode(".jpg", resized_frame)
        base64Frame=base64.b64encode(buffer).decode("utf-8")

        return base64Frame
    except:
        print('Frame',image_path,'not valid!!')
        return None
    

def image_paths_to_base64(image_paths):
    base64Frames = []
    
    for image_path in image_paths:
        # Read image
        image_path = image_path.strip()
        frame = cv2.imread(image_path)
        if frame is None:
            return False
            print(f"Warning: Could not read image {image_path}")
            continue
            
        # Resize image while maintaining aspect ratio
        height, width = frame.shape[:2]
        max_dimension = 768
        
        # Calculate new dimensions
        if height > width:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        else:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
            
        # Resize the image
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Encode as JPEG and convert to base64
        _, buffer = cv2.imencode(".jpg", resized_frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    
    return base64Frames




def _get_video_duration(video_path):
    """Get video duration in seconds using ffprobe"""
    with VideoFileClip(video_path) as clip:
        duration = clip.duration
    return duration

def _cut_video_clips(video_path, clips_dir, video_name, duration):    # Check if clips already exist
    existing_clips = [f for f in os.listdir(clips_dir) if f.endswith(".mp4")]
    expected_clips = int(duration // 10) + (1 if duration % 10 > 0 else 0)
    if len(existing_clips) >= expected_clips - 2:
        print(f"Skipping cutting: {len(existing_clips)} clips already exist in {clips_dir}")
        return

    """Cut video into 10-second clips using multithreading"""
    def cut_single_clip(start_time, end_time, clip_index):
        # Format time strings
        start_str = _seconds_to_time_str(start_time)
        end_str = _seconds_to_time_str(end_time)
        
        # Output filename
        output_file = os.path.join(
            clips_dir, 
            f"clip_{clip_index}_{start_str.replace(':', '-')}_to_{end_str.replace(':', '-')}.mp4"
        )
        
        # FFmpeg command to cut clip
        cmd = [
            'ffmpeg', '-i', video_path,
            '-ss', str(start_time),
            '-t', str(end_time - start_time),
            '-c', 'copy',  # Copy without re-encoding for speed
            '-avoid_negative_ts', 'make_zero',
            output_file,
            '-y'  # Overwrite if exists
        ]
        
        subprocess.run(cmd, capture_output=True)
        print(f"Created clip: {os.path.basename(output_file)}")
    
    # Calculate clip intervals
    clip_duration = 10  # seconds
    num_clips = int(duration // clip_duration) + (1 if duration % clip_duration > 0 else 0)
    
    # Use ThreadPoolExecutor for parallel clip cutting
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = []
        for i in range(num_clips):
            start_time = i * clip_duration
            end_time = min((i + 1) * clip_duration, duration)
            
            future = executor.submit(cut_single_clip, start_time, end_time, i)
            futures.append(future)
        
        # Wait for all clips to be processed
        for future in as_completed(futures):
            future.result()

def _seconds_to_time_str(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"




if __name__=='__main__':
    image_paths_to_base64(['./data/dense_frames/test_cartoon_18/frame_261.59.png'])

