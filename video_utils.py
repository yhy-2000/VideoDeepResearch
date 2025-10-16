import re
import json
import subprocess
from tqdm import tqdm
import os
import random
import time
import multiprocessing

from PIL import Image
import numpy as np
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import argparse
import random
import math
import os
from decord import VideoReader, cpu

try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
except:
    from moviepy import VideoFileClip, concatenate_videoclips
import os
import pandas as pd
from huggingface_hub import snapshot_download
import json
import cv2


MAX_FRAMES = 32
MIN_FRAMES = 20
MAX_RESOLUTION = 1024


from pathlib import Path

import ast
import re

def robust_eval(s: str):
    if not isinstance(s, str):
        return s
    
    # 原始尝试
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    
    # 预处理
    s_fixed = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    s_fixed = re.sub(r",\s*\.", ",", s_fixed)  # 修复 ", ." 为 ","
    s_fixed = re.sub(r"\)\s*\(", "),(", s_fixed)  # 修复 ") (" 为 "),("
    
    # 新增规则：处理嵌套括号不匹配的问题
    # 统计开括号和闭括号的数量
    open_paren_count = s_fixed.count('(')
    close_paren_count = s_fixed.count(')')
    
    # 如果开括号多于闭括号，补充缺失的闭括号
    if open_paren_count > close_paren_count:
        missing_closing = open_paren_count - close_paren_count
        s_fixed += ')' * missing_closing
    
    # 新增规则：处理字符串中未闭合的引号
    # 检查引号是否平衡
    double_quote_count = s_fixed.count('"')
    single_quote_count = s_fixed.count("'")
    
    # 如果双引号不平衡且为奇数，在字符串末尾添加闭合引号
    if double_quote_count % 2 == 1:
        s_fixed += '"'
    
    # 新增规则：处理字符串中的嵌套描述
    # 修复类似 "(100, 110, "描述..." 这样的嵌套字符串
    def fix_nested_strings(text):
        # 匹配未正确闭合的字符串模式
        pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*"([^"]*)$'
        
        def replacer(match):
            num1, num2, incomplete_desc = match.groups()
            # 简单修复：补全字符串和括号
            return f'({num1}, {num2}, "{incomplete_desc}")'
        
        # 应用修复
        fixed = re.sub(pattern, replacer, text)
        return fixed
    
    s_fixed = fix_nested_strings(s_fixed)
    
    # 新增规则：确保最外层是列表格式
    s_fixed = s_fixed.strip()
    if s_fixed.startswith("(") and not s_fixed.startswith("["):
        s_fixed = "[" + s_fixed + "]"
    
    # 最终尝试
    try:
        return ast.literal_eval(s_fixed)
    except Exception:
        # 如果还是失败，尝试更激进的修复
        try:
            # 移除所有多余的点和空格问题
            s_fixed = re.sub(r',\s*\.\s*,', ',', s_fixed)
            s_fixed = re.sub(r',\s*\.\s*\]', ']', s_fixed)
            s_fixed = re.sub(r',\s*\.\s*$', '', s_fixed)
            return ast.literal_eval(s_fixed)
        except Exception:
            return s_fixed
        

def custom_ffmpeg_extract_subclip(filename, t1, t2, targetname):
    flag, retry = 0, 0

    while flag != 1 and retry < 5:
        folder = os.path.dirname(targetname)
        os.makedirs(folder, exist_ok=True)

        try:
            video = VideoFileClip(filename)
            video_sub_clip = video.subclip(t1, t2)

            width, height = video.size
            scale = min(700 / max(width, height), 1)
            if scale < 1:
                video_sub_clip = video_sub_clip.resize(scale)

            video_sub_clip = concatenate_videoclips([video_sub_clip])
            video_sub_clip.write_videofile(targetname, codec="libx264", audio_codec="aac", logger=None)

            VideoReader(targetname, ctx=cpu(0), num_threads=1)
            flag = 1
        except Exception as e:
            print(f"Attempt {retry + 1} failed: {e}")
            retry += 1

    if not flag:
        print(f"Failed to process {targetname} after {retry} retries.")




def extract_uniform_frames(video_path, frame_path_folder, max_frames=32):
    
    os.makedirs(frame_path_folder, exist_ok=True)
    
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在 {video_path}")
        return None, None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return None, None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_list = []
    timestamp_list = []
    
    if total_frames <= max_frames:
        frame_indices = range(total_frames)
    else:
        frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
    
    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            timestamp = idx / original_fps if original_fps > 0 else idx * 0.2
            
            frame_filename = f"frame_{timestamp:.2f}.png"
            frame_path = os.path.join(frame_path_folder, frame_filename)
            
            success = cv2.imwrite(frame_path, frame)
            if success:
                frame_list.append(frame_path)
                timestamp_list.append(timestamp)
            else:
                print(f"  - 保存失败: {frame_path}")
    
    except Exception as e:
        print(f"处理视频时出错 {video_path}: {str(e)}")
        return None, None
    
    finally:
        cap.release()
    
    sorted_pairs = sorted(zip(timestamp_list, frame_list))
    timestamp_list, frame_list = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    print(f"完成处理: {Path(video_path).stem}, 共保存 {len(frame_list)} 帧 (目标: {max_frames} 帧)")
    return list(frame_list), list(timestamp_list)


import os
import cv2
from PIL import Image
def timestamp_to_frames(video_path, begin_timestamp, end_timestamp, frame_folder):
    
    max_frames = MAX_FRAMES
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    frame_folder = os.path.join(frame_folder, video_id)
    os.makedirs(frame_folder, exist_ok=True)
    
    clip_frames, timestamps = [], []
    for frame in sorted(os.listdir(frame_folder),key=lambda x:float(x.split('/')[-1].split('_')[-1].split('.')[0])):
        second = float(frame.split('/')[-1].split('_')[-1].split('.')[0])
        if second > end_timestamp:
            break
        if begin_timestamp <= second <= end_timestamp:
            clip_frames.append(f'{frame_folder}/{frame}')
            timestamps.append(second)
    
    if len(clip_frames) > max_frames:
        chunk_size = len(clip_frames) // max_frames
        clip_frames = clip_frames[::chunk_size][:max_frames]
        timestamps = timestamps[::chunk_size][:max_frames]
    return clip_frames, timestamps


def timestamp_to_clip_path(dataset_folder, begin_time_stamp, end_time_stamp, video_path, fps=2):
    max_frames = MAX_FRAMES
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    frame_folder = os.path.join(dataset_folder, f'dense_frames/{video_id}/')
    os.makedirs(frame_folder, exist_ok=True)

    if (end_time_stamp - begin_time_stamp) < 1:
        begin_time_stamp = max(begin_time_stamp - 0.5, 0)
        end_time_stamp += 0.5

    if int((end_time_stamp - begin_time_stamp) * fps) < max_frames: # densely
        num_frames = int((end_time_stamp - begin_time_stamp) * fps)
        time_points = [begin_time_stamp + i * (1.0 / fps) for i in range(num_frames)]
    else: # sparsely
        num_frames = max_frames
        time_points = [begin_time_stamp + i * ((end_time_stamp - begin_time_stamp) / num_frames) for i in range(num_frames)]
        
        
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_paths = []
    timestamps = []

    for t in time_points:
        frame_name = f"frame_{t:.2f}.png"
        frame_path = os.path.join(frame_folder, frame_name)
        
        valid_frame = False
        if os.path.exists(frame_path):
            try:
                with Image.open(frame_path) as img:
                    img.verify()
                valid_frame = True
            except:
                try:
                    os.remove(frame_path)
                except:
                    pass
        
        if not valid_frame:
            frame_idx = int(round(t * video_fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if success:
                cv2.imwrite(frame_path, frame)
                valid_frame = True
        
        if valid_frame:
            try:
                with Image.open(frame_path) as img:
                    img.verify()
                frame_paths.append(frame_path)
                timestamps.append(t)
            except:
                continue

    cap.release()

    if len(frame_paths) > max_frames:
        chunk_size = len(frame_paths) // max_frames
        frame_paths = frame_paths[::chunk_size][:max_frames]
        timestamps = timestamps[::chunk_size][:max_frames]

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
        

        for t in time_points:
            frame_name = f"frame_{t:.2f}.png"
            frame_path = os.path.join(frame_folder, frame_name)
            
            valid_frame = False
            
            if os.path.exists(frame_path):
                try:
                    with Image.open(frame_path) as img:
                        img.verify()
                    valid_frame = True
                except:
                    try:
                        os.remove(frame_path)
                    except:
                        pass
            
            if not valid_frame:
                frame_idx = int(round(t * video_fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = cap.read()
                if success:
                    cv2.imwrite(frame_path, frame)
                    valid_frame = True
            
            if valid_frame:
                try:
                    with Image.open(frame_path) as img:
                        img.verify()
                    frame_list.append(frame_path)
                    second_list.append(t)
                except Exception as e:
                    print(f"Error at time {t:.3f}: {e}")
                    continue

    cap.release()

    if len(frame_list) > MAX_FRAMES:
        interval = len(frame_list) / MAX_FRAMES
        frame_list = [frame_list[int(i * interval)] for i in range(MAX_FRAMES)]
        second_list = [second_list[int(i * interval)] for i in range(MAX_FRAMES)]

    if len(frame_list) < MIN_FRAMES and len(frame_list) > 0:
        interval = len(frame_list) / MIN_FRAMES
        frame_list = [frame_list[int(i * interval)] for i in range(MIN_FRAMES)]
        second_list = [second_list[int(i * interval)] for i in range(MIN_FRAMES)]

    if len(frame_list) == 0:
        raise KeyError(f'Frame list {dataset_folder} {clip_numbers} {video_path} {clip_duration} {fps} is invalid!')

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

from PIL import Image
from multiprocessing import Pool, cpu_count
import base64
import os
import cv2
import decord
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
from PIL import Image
import tempfile
from moviepy.video.io.VideoFileClip import VideoFileClip


MAX_DS_ROUND=20

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
    if ',' in s_ms:
        s, ms = s_ms.split(",")
    elif '.' in s_ms:
        s, ms = s_ms.split(".")
    else:
        s, ms = s_ms, 0
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


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
            img.verify()
        with Image.open(frame_path) as img:
            img.load()
        return True
    except:
        return False

import numpy as np
from PIL import Image
import base64
import os


def load_image(image_path, max_dimension_of_height_width=768):
    
    try:
        if not os.path.exists(image_path):
            print(f"文件不存在: {image_path}")
            return None
            
        frame = None
        
        try:
            pil_img = Image.open(image_path)
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            frame = np.array(pil_img)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"PIL读取失败 {image_path}: {e}")
        
        if frame is None:
            try:
                frame = cv2.imread(image_path)
                if frame is not None:
                    print(f"使用OpenCV成功读取: {image_path}")
            except Exception as e:
                print(f"OpenCV读取失败 {image_path}: {e}")
        
        if frame is None:
            try:
                with open(image_path, 'rb') as f:
                    img_data = np.frombuffer(f.read(), np.uint8)
                    frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if frame is not None:
                    print(f"使用二进制读取成功: {image_path}")
            except Exception as e:
                print(f"二进制读取失败 {image_path}: {e}")
                return None
        
        if frame is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        if frame.size == 0:
            raise ValueError(f"读取的图像为空: {image_path}")
            
        height, width = frame.shape[:2]
        
        if max(height, width) > max_dimension_of_height_width:
            max_dimension = max_dimension_of_height_width
            if height > width:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))
            else:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            resized_frame = frame
        
        success, buffer = cv2.imencode(".jpg", resized_frame, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            print(f"图片编码失败: {image_path}")
            return None
        
        base64Frame = base64.b64encode(buffer).decode("utf-8")
        return base64Frame
        
    except Exception as e:
        print(f"处理图像时出错 {image_path}: {e}")
        return None

def fix_corrupted_images(frame_folder):
    
    import glob
    from PIL import ImageFile
    
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    
    png_files = glob.glob(os.path.join(frame_folder, "*.png"))
    
    for png_file in png_files:
        try:
            with Image.open(png_file) as img:
                img.load()
                
                backup_file = png_file.replace('.png', '_backup.png')
                
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img.save(png_file, 'PNG', optimize=True)
                print(f"修复成功: {png_file}")
                
        except Exception as e:
            print(f"无法修复: {png_file}, 错误: {e}")
            
            try:
                os.remove(png_file)
                print(f"删除损坏文件: {png_file}")
            except:
                print(f"无法删除损坏文件: {png_file}")


def regenerate_frames_if_corrupted(video_path, frame_folder, max_frames=32):
    
    corrupted_files = []
    png_files = glob.glob(os.path.join(frame_folder, "*.png"))
    
    for png_file in png_files:
        try:
            with Image.open(png_file) as img:
                img.load()
        except Exception as e:
            print(f"发现损坏文件: {png_file}, 错误: {e}")
            corrupted_files.append(png_file)
    
    if corrupted_files:
        print(f"发现 {len(corrupted_files)} 个损坏文件，重新提取帧...")
        
        for png_file in png_files:
            try:
                os.remove(png_file)
            except:
                pass
        
        from pathlib import Path
        return extract_uniform_frames(video_path, frame_folder, max_frames)
    
    return None, None


def image_paths_to_base64(image_paths,  max_resolution=MAX_RESOLUTION):
    base64Frames = []
    
    for image_path in image_paths:
        image_path = image_path.strip()
        frame = cv2.imread(image_path)
        if frame is None:
            return False
            print(f"Warning: Could not read image {image_path}")
            continue
            
        height, width = frame.shape[:2]
        max_dimension = max_resolution
        
        if height > width:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        else:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
            
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        _, buffer = cv2.imencode(".jpg", resized_frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
    
    return base64Frames




def _get_video_duration(video_path):
    
    with VideoFileClip(video_path) as clip:
        duration = clip.duration
    return duration

def _cut_video_clips(video_path, clips_dir, video_name, duration):
    existing_clips = [f for f in os.listdir(clips_dir) if f.endswith(".mp4")]
    expected_clips = int(duration // 10) + (1 if duration % 10 > 0 else 0)
    if len(existing_clips) >= expected_clips - 2:
        print(f"Skipping cutting: {len(existing_clips)} clips already exist in {clips_dir}")
        return

    
    def cut_single_clip(start_time, end_time, clip_index):
        start_str = _seconds_to_time_str(start_time)
        end_str = _seconds_to_time_str(end_time)
        
        output_file = os.path.join(
            clips_dir, 
            f"clip_{clip_index}_{start_str.replace(':', '-')}_to_{end_str.replace(':', '-')}.mp4"
        )
        
        cmd = [
            'ffmpeg', '-i', video_path,
            '-ss', str(start_time),
            '-t', str(end_time - start_time),
            '-c', 'copy',
            '-avoid_negative_ts', 'make_zero',
            output_file,
            '-y' 
        ]
        
        subprocess.run(cmd, capture_output=True)
        print(f"Created clip: {os.path.basename(output_file)}")
    
    clip_duration = 10
    num_clips = int(duration // clip_duration) + (1 if duration % clip_duration > 0 else 0)
    
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = []
        for i in range(num_clips):
            start_time = i * clip_duration
            end_time = min((i + 1) * clip_duration, duration)
            
            future = executor.submit(cut_single_clip, start_time, end_time, i)
            futures.append(future)
        
        for future in as_completed(futures):
            future.result()

def _seconds_to_time_str(seconds):
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"




if __name__=='__main__':
    ans = load_image('./benchmark/mlvu/dense_frames/cXDT44zT8JY/frame_336.34.png')
