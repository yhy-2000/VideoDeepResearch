"""
视频问答系统 Demo - 完整多轮工具调用版本
处理单个视频和问题，包含完整的工具调用流程
"""

import os
import sys
import json
import re
import time
import torch
from pathlib import Path
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from PIL import Image
from openai import OpenAI
import random

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# 导入必要的工具函数
from video_utils import (
    timestamp_to_clip_path, extract_subtitles, parse_subtitle_time,
    timestamp_to_frames, extract_video_clip, robust_eval
)
from retriever_languagebind import Retrieval_Manager
from prompt import (
    initial_input_template_subtitle, 
    initial_input_template_wo_subtitle,
    initial_input_template_temporal_grounding_agent,
    initial_input_template_temporal_grounding_agent_wo_subtitle
)

import os
from PIL import Image
import io
from multiprocessing import Pool, cpu_count
from functools import partial
import multiprocessing as mp
import hashlib
import json
import os
import time
import pickle
import fcntl
from pathlib import Path

def safe_write_with_lock(data, file_path):
    
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'wb') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            pickle.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def safe_read_with_lock(file_path):
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'rb') as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            try:
                return pickle.load(f)
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
    except (EOFError, pickle.UnpicklingError, OSError) as e:
        print(f'读取文件时出错 {file_path}: {e}')
        return None

def list_to_sha256(lst):
    json_str = json.dumps(lst, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


MAX_DS_ROUND = 20  # 最大对话轮数

class VideoQADemo:
    """单视频问答处理Demo类 - 完整工具调用版本"""
    
    def __init__(self, 
                 video_path: str,
                 question: str,
                 answer: str = None,
                 options: list = None,
                 dataset_folder: str = "./data",
                 clip_duration: int = 10,
                 use_subtitle: bool = True,
                 vlm_model_name: str = None,
                 planner_model_name: str = None,
                 temporal_model_name: str = None):
        """
        初始化Demo
        
        Args:
            video_path: 视频文件路径
            question: 问题文本
            answer: 正确答案（用于评估）
            options: 选项列表（可选）
            dataset_folder: 数据集文件夹
            clip_duration: 视频片段时长（秒）
            use_subtitle: 是否使用字幕
            vlm_model_name: VLM模型名称
            planner_model_name: 规划模型名称
            temporal_model_name: 时序定位模型名称
        """
        self.video_path = video_path
        self.question = question
        self.answer = answer
        self.options = options or []
        self.dataset_folder = dataset_folder
        self.clip_duration = clip_duration
        self.use_subtitle = use_subtitle
        
        # 设置环境变量
        self._setup_environment()
        
        # 初始化模型名称
        self.vlm_model_name = vlm_model_name or os.getenv('API_MODEL_NAME_VLM', 'Qwen/Qwen2-VL-7B-Instruct')
        self.planner_model_name = planner_model_name or os.getenv('API_MODEL_NAME', 'deepseek-ai/DeepSeek-V3')
        self.temporal_model_name = temporal_model_name or os.getenv('API_MODEL_NAME_TEMPORAL_GROUNDING', 'deepseek-ai/DeepSeek-V3')
        
        # 初始化API配置
        self._setup_api_config()
        
        # 初始化模型
        self._initialize_models()
        
        # 获取视频时长
        self.duration = self._get_video_duration()
        
        # 初始化检索器
        self.retriever = self._initialize_retriever()
        
        # 提取字幕
        self.subtitles = self._extract_subtitles()
        
        # 对话历史
        self.messages = []
        
        print(f"✓ Demo initialized successfully")
        print(f"  Video: {video_path}")
        print(f"  Duration: {self.duration}s")
        print(f"  Question: {question}")
        if self.subtitles:
            print(f"  Subtitles: {len(self.subtitles)} characters")
    
    def _setup_environment(self):
        """设置环境变量"""
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        os.environ.setdefault("VLLM_USE_MODELSCOPE", "false")
        torch.backends.cuda.matmul.allow_tf32 = True
    
    def _setup_api_config(self):
        """设置API配置"""
        # 规划模型API配置
        self.planner_api_base = os.getenv('API_BASE_URL', 'http://localhost:8000/v1').split(',')
        self.planner_api_keys = os.getenv('API_KEY', 'EMPTY').split(',')
        
        # 时序定位模型API配置
        self.temporal_api_base = os.getenv('API_BASE_URL_TEMPORAL_GROUNDING', 'http://localhost:8001/v1').split(',')
        self.temporal_api_keys = os.getenv('API_KEY_TEMPORAL_GROUNDING', 'EMPTY').split(',')
    
    def _initialize_models(self):
        """初始化VLM模型"""
        print("Initializing VLM model...")
        
        self.vlm_server = LLM(
            model=self.vlm_model_name,
            gpu_memory_utilization=0.85,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=32768,
            enable_chunked_prefill=True,
            enforce_eager=True,
        )
        
        self.processor = AutoProcessor.from_pretrained(
            self.vlm_model_name, 
            use_fast=True
        )
        self.processor.tokenizer.padding_side = 'left'
        
        print(f"✓ VLM model loaded: {self.vlm_model_name}")
    
    def _initialize_retriever(self):
        """初始化检索器"""
        print("Initializing retriever...")
        
        # 创建临时args对象
        class Args:
            dataset_folder = self.dataset_folder
            dataset = "demo"
            clip_duration = self.clip_duration
            retriever_type = "large"
        
        args = Args()
        clip_save_folder = f'{self.dataset_folder}/clips/{self.clip_duration}/'
        
        retriever = Retrieval_Manager(args, clip_save_folder=clip_save_folder)
        
        # 如果有GPU，加载到GPU
        if torch.cuda.is_available():
            retriever.load_model_to_gpu(0)
        
        print(f"✓ Retriever initialized")
        return retriever
    
    def _get_video_duration(self):
        """获取视频时长"""
        try:
            from moviepy.video.io.VideoFileClip import VideoFileClip
            with VideoFileClip(self.video_path) as video:
                return int(video.duration)
        except Exception as e:
            print(f"Warning: Could not get video duration: {e}")
            return 300  # 默认5分钟
    
    def _extract_subtitles(self):
        """提取字幕"""
        if not self.use_subtitle:
            return ""
        
        video_id = Path(self.video_path).stem
        subtitle_path = f'{self.dataset_folder}/subtitles/{video_id}.srt'
        
        if not os.path.exists(subtitle_path):
            print("No subtitle file found")
            return ""
        
        subtitles = ""
        try:
            with open(subtitle_path, "r", encoding="utf-8") as f:
                content = f.read().split("\n\n")
                for section in content:
                    if section.strip():
                        lines = section.split("\n")
                        if len(lines) >= 3:
                            time_range = lines[1].split(" --> ")
                            start_time = parse_subtitle_time(time_range[0])
                            end_time = parse_subtitle_time(time_range[1])
                            text = " ".join(lines[2:])
                            subtitles += f"{int(start_time)}-{int(end_time)}:{text} "
        except Exception as e:
            print(f"Error extracting subtitles: {e}")
            return ""
        
        return subtitles
    
    def _build_initial_prompt(self):
        """构建初始提示"""
        options_text = "\n".join(self.options) if self.options else ""
        question_text = self.question + "\n" + options_text
        
        if self.use_subtitle:
            prompt = initial_input_template_subtitle.format(
                question=question_text,
                duration=self.duration,
                clip_duration=self.clip_duration,
                MAX_DS_ROUND=MAX_DS_ROUND
            )
        else:
            prompt = initial_input_template_wo_subtitle.format(
                question=question_text,
                duration=self.duration,
                clip_duration=self.clip_duration,
                MAX_DS_ROUND=MAX_DS_ROUND
            )
        
        return prompt.replace('thinking>', 'think>')
    
    def _text2text(self, message: list, model_name: str, api_base: list, api_keys: list) -> str:
        
        folder_path = '_temporal' if 'temporal' in model_name else '_planner'
        start_time = time.time()

        index = message + [model_name] + [len(message)]
        file_name = f'{list_to_sha256(index)}.pkl'
        read_file = f'./vllm_io_files/vllm_input{folder_path}/{file_name}'
        safe_write_with_lock({'model': model_name, 'input': message},read_file)
        start_time = time.time()
        while True:
            output_file = f'./vllm_io_files/vllm_output{folder_path}/{file_name}'
            if os.path.exists(output_file):
                end_time = time.time()
                try:
                    ans = safe_read_with_lock(output_file)
                    return ans
                except Exception as e:
                    print('[TEXT2TEXT] ERROR:', e)
                    safe_write_with_lock({'model': model_name, 'input': message},read_file)
                    os.system(f'rm {output_file}')

            if time.time()-start_time>120:
                break
            if not os.path.exists(read_file):
                safe_write_with_lock({'model': model_name, 'input': message},read_file)
            time.sleep(0.2)

        print('[TEXT2TEXT] ERROR: Timeout, model:', model_name)
        return ''
    
    
    def _batch_video2text(self, tasks: list):
        """批量处理视频片段"""
        results = []
        
        for prompt, image_paths, timestamps in tasks:
            # 加载图像
            image_data = []
            for img_path in image_paths:
                if os.path.exists(img_path):
                    try:
                        image = Image.open(img_path)
                        image.verify()
                        image = Image.open(img_path)
                        
                        # 调整大小
                        width, height = image.size
                        if max(width, height) > 768:
                            if width > height:
                                new_width = 768
                                new_height = int(height * (768 / width))
                            else:
                                new_height = 768
                                new_width = int(width * (768 / height))
                            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        
                        image_data.append(image)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
                        continue
            
            if not image_data:
                results.append("Error: No valid frames")
                continue
            
            # 构建消息
            content = [
                {"type": "video", "video": image_paths},
                {"type": "text", "text": prompt}
            ]
            messages = [{"role": "user", "content": content}]
            
            # 格式化提示
            formatted_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 生成
            sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
            fps = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 2.0
            
            outputs = self.vlm_server.generate(
                {
                    "prompt": formatted_prompt,
                    "multi_modal_data": {"video": image_data},
                    "mm_processor_kwargs": {
                        "min_pixels": 4 * 28 * 28,
                        "max_pixels": 768 * 28 * 28,
                        "fps": fps,
                    },
                },
                sampling_params=sampling_params,
                use_tqdm=False
            )
            
            result = outputs[0].outputs[0].text.strip()
            results.append(result)
        
        return results
    
    def _extract_final_answer(self, text: str) -> str:
        """提取最终答案"""
        try:
            answer_content = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)[-1].strip()
            first_upper = re.search(r'[A-Z]', answer_content)
            return first_upper.group(0) if first_upper else answer_content
        except:
            return '-'
    
    # ==================== 工具处理函数 ====================
    
    def _process_temporal_grounding(self, output_text: str) -> str:
        """处理时序定位代理"""
        print("\n[Tool] Temporal Grounding Agent")
        
        pattern = r"<temporal_grounding_agent>([^<]+)</temporal_grounding_agent>"
        try:
            question = re.findall(pattern, output_text)[0]
        except:
            print("Warning: No valid temporal_grounding_agent found")
            return ""
        
        # 构建代理初始提示
        if self.use_subtitle:
            agent_prompt = initial_input_template_temporal_grounding_agent.format(
                clip_duration=10, question=question, duration=self.duration
            )
        else:
            agent_prompt = initial_input_template_temporal_grounding_agent_wo_subtitle.format(
                clip_duration=10, question=question, duration=self.duration
            )
        
        agent_prompt = agent_prompt.replace('thinking>', 'think>')
        agent_messages = [{"role": "user", "content": agent_prompt}]
        
        # 第一轮：检索视频片段
        tool_call = self._text2text(agent_messages, self.temporal_model_name, 
                                    self.temporal_api_base, self.temporal_api_keys)
        agent_messages.append({"role": "assistant", "content": tool_call})
        
        # 处理检索结果
        tool_results = self._process_tool_calls_for_temporal(tool_call)
        agent_messages.append({
            "role": "user",
            "content": tool_results + "\nNow you should call the video reader to check the video segments."
        })
        
        # 第二轮：验证视频片段
        tool_call = self._text2text(agent_messages, self.temporal_model_name,
                                    self.temporal_api_base, self.temporal_api_keys)
        agent_messages.append({"role": "assistant", "content": tool_call})
        
        # 处理验证结果
        tool_results = self._process_tool_calls_for_temporal(tool_call)
        agent_messages.append({
            "role": "user",
            "content": tool_results + "\nNow you should output the final video segments."
        })
        
        # 第三轮：输出最终结果
        tool_call = self._text2text(agent_messages, self.temporal_model_name,
                                    self.temporal_api_base, self.temporal_api_keys)
        
        # 提取答案
        if '<answer>' in tool_call:
            answer_text = re.findall(r'<answer>([^<]+)</answer>', tool_call, re.DOTALL)[-1].strip()
            b_idx, e_idx = answer_text.find('['), answer_text.find(']')
            if b_idx != -1 and e_idx != -1:
                answer_text = answer_text[b_idx:e_idx+1]
                intervals = robust_eval(answer_text)
                result = f'There are {len(intervals)} related segments in the video: {intervals}'
                print(f"  Found {len(intervals)} segments")
                return result
        
        return "No segments found"
    
    def _process_video_reader(self, output_text: str) -> str:
        """处理视频阅读器工具"""
        print("\n[Tool] Video Reader")
        
        pattern = r"<video_reader>([^<]+)</video_reader>\s*<video_reader_question>([^<]+)</video_reader_question>"
        matches = re.findall(pattern, output_text.strip())
        
        if not matches:
            return ""
        
        tasks = []
        time_matches = [match[0] for match in matches]
        question_matches = [match[1] for match in matches]
        
        for query, time_match in zip(question_matches, time_matches):
            begin_time, end_time = time_match.split(':')
            begin_time, end_time = float(begin_time), float(end_time)
            
            video_clip, timestamps = timestamp_to_clip_path(
                self.dataset_folder, begin_time, end_time, 
                self.video_path, fps=2.0
            )
            
            if len(video_clip) > 0:
                query_formatted = (
                    f"Please watch the given video and answer the following question: {query}\n"
                    "Output the detailed video description and the answer in this format: "
                    "The description of the video is:YOUR_DESCRIPTION\nThe answer is:YOUR_ANSWER."
                )
                tasks.append((query_formatted, video_clip, timestamps))
        
        if not tasks:
            return ""
        
        results = self._batch_video2text(tasks)
        
        ans = ""
        for time_match, result in zip(time_matches, results):
            ans += f'The tool result for <video_reader>{time_match}</video_reader> is {result}\n'
            print(f"  Processed segment: {time_match}")
        
        return ans
    
    def _process_video_segment_retriever_text(self, output_text: str) -> str:
        """处理基于文本的视频片段检索"""
        print("\n[Tool] Video Segment Retriever (Text)")
        
        pattern = r"<video_segment_retriever_textual_query>(.*?)</video_segment_retriever_textual_query>"
        matches = re.findall(pattern, output_text, flags=re.DOTALL)
        
        if not matches:
            return ""
        
        results = []
        topk = int(os.getenv('TOPK', '5'))
        
        for match in matches:
            for query in match.split(';'):
                try:
                    video_clip_paths = self.retriever.get_informative_clips(
                        query, video_path=self.video_path, 
                        top_k=topk, total_duration=self.duration
                    )
                    cur_video_paths = [
                        int(video[0].split('/')[-1].split('_')[1]) 
                        for video in video_clip_paths
                    ]
                    results.append(
                        f"The tool results for <video_segment_retriever_textual_query>{query}"
                        f"</video_segment_retriever_textual_query> are:\n{cur_video_paths}\n"
                    )
                    print(f"  Query: {query[:50]}... -> Found {len(cur_video_paths)} clips")
                except Exception as e:
                    print(f"  Error: {e}")
                    continue
        
        return ''.join(results)
    
    def _process_subtitle_retriever(self, output_text: str) -> str:
        """处理字幕检索器"""
        print("\n[Tool] Subtitle Retriever")
        
        pattern = r"<subtitle_retriever>(.*?)</subtitle_retriever>"
        matches = re.findall(pattern, output_text, flags=re.DOTALL)
        
        if not matches:
            return ""
        
        results = []
        topk = 10
        
        for match in matches:
            subtitle_triples = []
            vis = []
            
            for query in match.split(';'):
                try:
                    cur_subtitle_triples = self.retriever.get_informative_subtitles(
                        query, video_path=self.video_path,
                        top_k=topk, total_duration=self.duration
                    )
                    
                    for x in cur_subtitle_triples:
                        if x[0] not in vis:
                            subtitle_triples.append({
                                'begin_timestamp': x[0],
                                'end_timestamp': x[1],
                                'text': x[2]
                            })
                            vis.append(x[0])
                except Exception as e:
                    print(f"  Error: {e}")
                    continue
            
            subtitle_triples = sorted(subtitle_triples, key=lambda x: x['begin_timestamp'])
            results.append(
                f"The tool results for <subtitle_retriever>{match}</subtitle_retriever> are:\n"
                f"{subtitle_triples}\n"
            )
            print(f"  Found {len(subtitle_triples)} subtitle segments")
        
        return ''.join(results)
    
    def _process_video_browser(self, output_text: str) -> str:
        """处理视频浏览器"""
        print("\n[Tool] Video Browser")
        
        pattern = r"<video_browser>([^<]+)</video_browser>"
        queries = re.findall(pattern, output_text)
        
        if not queries:
            return ""
        
        query = queries[0]
        video_clip, timestamps = timestamp_to_clip_path(
            self.dataset_folder, 0, self.duration, 
            self.video_path, fps=2.0
        )
        
        ans = self._batch_video2text([(query, video_clip, timestamps)])[0]
        print(f"  Browsed entire video")
        return f"The tool results for <video_browser>{query}</video_browser> is:{ans}\n"
    
    def _process_tool_calls(self, output_text: str) -> str:
        """处理工具调用（主循环）"""
        tool_result = ""
        
        if "<temporal_grounding_agent>" in output_text:
            tool_result += self._process_temporal_grounding(output_text)
        
        if "<video_reader>" in output_text:
            tool_result += self._process_video_reader(output_text)
        
        if '<video_segment_retriever_textual_query>' in output_text:
            tool_result += self._process_video_segment_retriever_text(output_text)
        
        if '<subtitle_retriever>' in output_text:
            tool_result += self._process_subtitle_retriever(output_text)
        
        if "<video_browser>" in output_text:
            tool_result += self._process_video_browser(output_text)
        
        return tool_result
    
    def _process_tool_calls_for_temporal(self, output_text: str) -> str:
        """处理时序定位代理内部的工具调用"""
        tool_result = ""
        
        if "<video_reader>" in output_text:
            tool_result += self._process_video_reader(output_text)
        
        if '<video_segment_retriever_textual_query>' in output_text:
            tool_result += self._process_video_segment_retriever_text(output_text)
        
        if '<subtitle_retriever>' in output_text:
            tool_result += self._process_subtitle_retriever(output_text)
        
        return tool_result
    
    def run(self):
        """运行完整的多轮工具调用流程"""
        print("\n" + "="*70)
        print("Starting Video QA Demo - Multi-Turn Tool Calling")
        print("="*70 + "\n")
        
        # 构建初始提示
        initial_prompt = self._build_initial_prompt()
        self.messages = [{
            "role": "user",
            "content": [{"type": "text", "text": initial_prompt}]
        }]
        
        cur_turn = 0
        
        # 多轮对话循环
        while cur_turn < MAX_DS_ROUND:
            print(f"\n{'='*70}")
            print(f"Round {cur_turn + 1}/{MAX_DS_ROUND}")
            print(f"{'='*70}")
            
            # 调用规划模型
            print("\n[Planner] Generating response...")
            output_text = self._text2text(
                self.messages, 
                self.planner_model_name,
                self.planner_api_base,
                self.planner_api_keys
            )
            print(f"\n[Planner Output]\n{output_text}\n")
            
            if not output_text:
                print("Error: No response from planner")
                break
            
            # 记录规划器输出
            self.messages.append({'role': 'assistant', 'content': output_text})
            cur_turn += 1
            
            # 检查是否有最终答案
            if '<answer>' in output_text:
                answer = self._extract_final_answer(output_text)
                print(f"\n{'='*70}")
                print(f"Final Answer: {answer}")
                print(f"{'='*70}\n")
                
                # 评估准确性
                is_correct = False
                if self.answer:
                    is_correct = (answer == self.answer)
                    print(f"Ground Truth: {self.answer}")
                    print(f"Correctness: {'✓ Correct' if is_correct else '✗ Incorrect'}\n")
                
                return {
                    'messages': self.messages,
                    'pred_answer': answer,
                    'ground_truth': self.answer,
                    'is_correct': is_correct,
                    'total_rounds': cur_turn
                }
            
            # 处理工具调用
            print("\n[Tool Processor] Processing tool calls...")
            tool_result = self._process_tool_calls(output_text)
            print(f"\n[Tool Results]\n{tool_result}\n")
            
            if tool_result:
                self.messages.append({
                    'role': 'user',
                    'content': tool_result + f"\nYou have now engaged in {cur_turn} rounds of conversation, "
                               f"with {MAX_DS_ROUND-cur_turn} calls remaining."
                })
                print(f"\n[System] Tool results provided to planner")
            elif '<answer>' not in output_text:
                self.messages.append({
                    'role': 'user',
                    'content': 'The output is invalid. You should strictly follow the provided xml format!!!'
                })
                print("\n[System] Warning: Invalid output format")
            
            # 检查是否达到最大轮数
            if cur_turn >= MAX_DS_ROUND:
                self.messages.append({
                    'role': 'user',
                    'content': 'Maximum number of rounds reached! Now you should output the final answer within <answer></answer>!!!'
                })
                print("\n[System] Maximum rounds reached, forcing answer...")
        
        print(f"\n{'='*70}")
        print("Demo completed (max rounds reached without answer)")
        print(f"{'='*70}\n")
        
        return {
            'messages': self.messages,
            'pred_answer': '-',
            'ground_truth': self.answer,
            'is_correct': False,
            'total_rounds': cur_turn
        }


def main():
    """主函数 - 运行Demo"""
    
    # ==================== 配置参数 ====================
    VIDEO_PATH = "/share/project/huaying/benchmark/mlvu/videos/test_tutorial_21.mp4"  # replace the video_path to any video you want to test
    QUESTION = "Which technique is primarily used when editing the landscape photo in the video?"
    ANSWER = "C"  
    OPTIONS = [
        "A. Sharpening",
        "B. Color correction",
        "C. Compositing",
        "D. Contrast enhancement",
        "E. Saturation adjustment",
        "F. Exposure adjustment"
    ]
    
    # 检查视频是否存在
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found: {VIDEO_PATH}")
        print("Please update VIDEO_PATH in the script to point to a valid video file.")
        return
    
    # ==================== 创建并运行Demo ====================
    demo = VideoQADemo(
        video_path=VIDEO_PATH,
        question=QUESTION,
        answer=ANSWER,
        options=OPTIONS,
        dataset_folder="./data",
        clip_duration=10,
        use_subtitle=False,  # 如果没有字幕文件，设为False
    )
    
    # 运行完整流程
    result = demo.run()
    
    # ==================== 保存结果 ====================
    output_path = "demo_result.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'video_path': VIDEO_PATH,
            'question': QUESTION,
            'options': OPTIONS,
            'ground_truth': ANSWER,
            'predicted_answer': result['pred_answer'],
            'is_correct': result['is_correct'],
            'total_rounds': result['total_rounds'],
            'conversation_length': len(result['messages'])
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # ==================== 打印统计信息 ====================
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Question: {QUESTION}")
    print(f"Predicted Answer: {result['pred_answer']}")
    if ANSWER:
        print(f"Ground Truth: {ANSWER}")
        print(f"Correctness: {'✓ Correct' if result['is_correct'] else '✗ Incorrect'}")
    print(f"Total Rounds: {result['total_rounds']}")
    print(f"Conversation Length: {len(result['messages'])} messages")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()