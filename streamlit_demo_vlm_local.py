import os
os.environ["VLLM_USE_MODELSCOPE"] = "false"   
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor 
from video_utils import _get_video_duration,_cut_video_clips,extract_subtitles,timestamp_to_clip_path,is_valid_video,is_valid_frame,extract_video_clip,parse_subtitle_time,clip_number_to_clip_path,image_paths_to_base64
import json
import re
import torch
import decord
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Any, Callable, Optional
from decord import VideoReader, cpu
from PIL import Image
from pathlib import Path
import argparse
from retriever import Retrieval_Manager
from prompt import *
from collections import defaultdict
import random
from qwen_vl_utils import process_vision_info
import time
from vllm import LLM, EngineArgs, SamplingParams
from openai import OpenAI
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from moviepy.video.io.VideoFileClip import VideoFileClip
import streamlit as st

# 环境配置
os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.backends.cuda.matmul.allow_tf32 = True

MAX_DS_ROUND=20


# 在StreamlitVideoQAManager类之前添加示例配置
EXAMPLE_CONFIGS = [
    {
        "name": "🐭 Cartoon Detail Anaysis",
        "question": "What did the mouse take from the box after running into the mouse hole? A. Knife B. White flag C. Sword D. Cheese E. Apple F. Stick",
        "video_path": "./data/videos/test_cartoon_18.mp4",
        "subtitle_path": "",
        "description": "Plot and detail analysis",
        "category": "Animation",
        "duration": "~6 minutes"
    },
    {
        "name": "📽️ Plot Reasoning",
        "question": "Why does the man follow the woman in red after they meet for the third time?(A) He falls in love with the woman(B) He is a spy(C) The woman has his phone(D) The woman has his book",
        "video_path": "./data/videos/q01CUy_gwdU.mp4",
        "subtitle_path": "",
        "description": "Plot Reasoning",
        "category": "Plot",
        "duration": "~55 minutes"
    },
    {
        "name": "🏊🏻 Sport Temporal Reasoning",
        "question": "What is the halftime score for Team USA vs Team Australia? (A) 28:21 (B) 42:56 (C) 21:28 (D) 56:42",
        "video_path": "./data/videos/Aiem1w_TvaA.mp4",
        "subtitle_path": "",
        "description": "Sport Temporal Reasoning",
        "category": "Sport",
        "duration": "~2h"
    },
    {
        "name": "🧍‍♂️Event Understanding",
        "question": "When the Best Supporting Actor winner goes offstage, what is the color of the outfit of the person sitting next to him?\n(A) Blue\n(B) White\n(C) Black\n(D) Red",
        "video_path": "./data/videos/rk24OUu_kJQ.mp4",
        "subtitle_path": "",
        "description": "Event Understanding",
        "category": "",
        "duration": ""
    } 
]



def create_examples_section():
    """创建示例选择区域"""
    st.markdown("### 🎯 Quick Start Examples")
    st.markdown("Choose from pre-configured examples or create your own:")
    
    # 创建示例卡片
    cols = st.columns(2)
    
    for i, example in enumerate(EXAMPLE_CONFIGS):
        with cols[i % 2]:
            with st.container():
                # 创建示例卡片
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 1.5rem;
                    border-radius: 15px;
                    margin: 1rem 0;
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                    border: 1px solid rgba(255,255,255,0.1);
                ">
                    <h4 style="margin-bottom: 0.5rem;">{example['name']}</h4>
                    <div style="margin-bottom: 0.5rem;">
                        <span style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.5rem; border-radius: 10px; font-size: 0.8rem;">
                            {example['category']}
                        </span>
                        <span style="background: rgba(255,255,255,0.2); padding: 0.2rem 0.5rem; border-radius: 10px; font-size: 0.8rem; margin-left: 0.5rem;">
                            {example['duration']}
                        </span>
                    </div>
                    <p style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 1rem;">{example['description']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # 添加使用按钮
                if st.button(
                    f"📋 Use Example {i+1}", 
                    key=f"example_{i}",
                    use_container_width=True,
                    help=f"Load: {example['name']}"
                ):
                    # 将示例数据存储到session state
                    st.session_state.selected_example = example
                    st.session_state.example_loaded = True
                    st.success(f"✅ Loaded: {example['name']}")
                    st.rerun()

def get_example_values():
    """获取当前选中示例的值"""
    if 'selected_example' in st.session_state and st.session_state.get('example_loaded', False):
        example = st.session_state.selected_example
        return {
            'question': example['question'],
            'video_path': example['video_path'], 
            'subtitle_path': example['subtitle_path']
        }
    return {
        'question': '',
        'video_path': '',
        'subtitle_path': ''
    }


def create_enhanced_demo():
    """创建增强版的Streamlit演示"""
    st.set_page_config(
        page_title="🎬 Video DeepResearch Demo",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 样式部分
    st.markdown("""<style>...[你的 CSS 样式保持不变，此处省略]...</style>""", unsafe_allow_html=True)

    st.markdown('<div class="main-header">🎬 Enhanced Video QA Demo</div>', unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("#### 🤖 Model Settings")
        clip_duration = st.slider("Clip Duration (s)", 5, 30, 10)
        clip_fps = st.slider("Clip FPS", 1.0, 10.0, 3.0, 0.5)
        use_subtitle = st.checkbox("Use Subtitles", False)
        with st.expander("🔧 Advanced Settings"):
            max_rounds = st.slider("Max Processing Rounds", 5, 30, 20)
            enable_debug = st.checkbox("Enable Debug Mode", False)
            auto_save = st.checkbox("Auto Save Results", True)

        st.markdown("#### 📊 System Status")
        gpu_available = torch.cuda.is_available() if 'torch' in globals() else False
        st.info(f"🎮 GPU: {'Available' if gpu_available else 'Not Available'}")
        st.info(f"🧠 Model: DeepSeek-R1")
        st.info(f"👁 VLM: Qwen2.5-VL-7B-Instruct")
        if gpu_available:
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                st.info(f"💾 GPU Memory: {gpu_memory:.1f} GB")
            except:
                pass

    tab1, tab2, tab3 = st.tabs(["🎯 Main Processing", "📊 Analytics", "⚙️ Settings"])


    def display_examples_above_input():
        st.markdown("### 📚 Choose an Example")

        num_cols = 5  # 一行5个
        cols = st.columns(num_cols)

        for idx, ex in enumerate(EXAMPLE_CONFIGS):
            col = cols[idx % num_cols]

            card_html = f"""
            <div style="
                border: 1px solid #ddd; 
                border-radius: 8px; 
                padding: 8px 10px; 
                margin-bottom: 8px;
                box-shadow: 1px 1px 3px rgba(0,0,0,0.07);
                background: #fafafa;
                font-size: 0.85em;
                cursor: pointer;
                height: 110px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                ">
                <div style="font-weight: 600; line-height: 1.1; min-height: 36px; overflow: hidden;">{ex['name']}</div>
                <div style="color: #555; font-size: 0.75em; flex-grow: 1; overflow: hidden;">{ex['description'] or 'No description'}</div>
                <small style="color: #888; font-size: 0.7em;">{ex['category']} · {ex['duration']}</small>
            </div>
            """

            col.markdown(card_html, unsafe_allow_html=True)

            if col.button("Select", key=f"load_example_{idx}", help=f"Load example: {ex['name']}"):
                st.session_state.selected_example = ex
                st.session_state.example_loaded = True
                st.rerun()


    def get_example_values():
        if st.session_state.get('example_loaded', False):
            return {
                'question': st.session_state['selected_example']['question'],
                'video_path': st.session_state['selected_example']['video_path'],
                'subtitle_path': st.session_state['selected_example'].get('subtitle_path', '')
            }
        else:
            return {
                'question': '',
                'video_path': '',
                'subtitle_path': ''
            }

    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            display_examples_above_input()  # 🔄 显示所有示例

            st.markdown("### 📝 Input Configuration")
            example_values = get_example_values()

            # 显示当前加载的示例
            if st.session_state.get('example_loaded', False):
                example = st.session_state.selected_example
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                            color: white; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                    <strong>📋 Loaded Example:</strong> {example['name']}<br>
                    <small>{example['description']}</small>
                </div>
                """, unsafe_allow_html=True)
                if st.button("🗑️ Clear Example", help="Clear the loaded example"):
                    st.session_state.example_loaded = False
                    if 'selected_example' in st.session_state:
                        del st.session_state.selected_example
                    st.rerun()

            with st.form("enhanced_qa_form"):
                question = st.text_area(
                    "❓ Your Question",
                    value=example_values['question'],
                    placeholder="What happens in this video? Please describe the main events...",
                    height=120
                )

                video_path = st.text_input(
                    "🎥 Video File Path",
                    value=example_values['video_path'],
                    placeholder="./data/videos/example.mp4"
                )

                subtitle_path = st.text_input(
                    "📄 Subtitle File Path (Optional)",
                    value=example_values['subtitle_path'],
                    placeholder="./data/subtitles/example.srt"
                )

                col_btn1, col_btn2, col_btn3 = st.columns(3)
                with col_btn1:
                    submitted = st.form_submit_button("🚀 Start Processing", use_container_width=True)
                with col_btn2:
                    if st.form_submit_button("🔍 Validate Inputs", use_container_width=True):
                        if video_path and os.path.exists(video_path):
                            st.success("✅ Video file found!")
                        else:
                            st.error("❌ Video file not found!")
                        if subtitle_path:
                            if os.path.exists(subtitle_path):
                                st.success("✅ Subtitle file found!")
                            else:
                                st.warning("⚠️ Subtitle file not found!")
                with col_btn3:
                    if st.form_submit_button("📚 Browse Examples", use_container_width=True):
                        st.info("👆 Now shown directly above input section!")

        with col2:
            if video_path and os.path.exists(video_path):
                st.markdown("### 📹 Video Preview")
                try:
                    st.video(video_path)
                    with VideoFileClip(video_path) as clip:
                        duration = clip.duration
                        fps = clip.fps
                        size = clip.size
                    st.markdown(f"""
                    <div class="stats-container">
                        <h4>📊 Video Information</h4>
                        <div><strong>Duration:</strong> {duration:.2f} seconds</div>
                        <div><strong>FPS:</strong> {fps}</div>
                        <div><strong>Resolution:</strong> {size[0]}x{size[1]}</div>
                        <div><strong>File Size:</strong> {os.path.getsize(video_path) / (1024*1024):.2f} MB</div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error loading video: {e}")
            else:
                st.info("👆 Please enter a valid video path to see preview")

        if submitted and question and video_path:
            if os.path.exists(video_path):
                args = argparse.Namespace()
                args.video_path = video_path
                args.subtitle_path = subtitle_path
                args.use_subtitle = use_subtitle
                args.clip_duration = clip_duration
                args.clip_fps = clip_fps
                args.dataset_folder = './data'
                args.mllm_path = "Qwen/Qwen2.5-VL-7B-Instruct"

                try:
                    with st.spinner("🔄 Initializing Video QA Manager..."):
                        manager = StreamlitVideoQAManager(args)
                    st.success("✅ Manager initialized successfully!")

                    st.markdown("### 🔄 Processing in Progress")
                    result = manager.process_single_input_with_ui(question, video_path, subtitle_path)

                    if result:
                        st.markdown(f"""
                        <div class="result-card">
                            <h2>🎯 Final Answer</h2>
                            <h3>{result['answer']}</h3>
                        </div>
                        """, unsafe_allow_html=True)

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🔄 Rounds", result.get('rounds', 0))
                        with col2:
                            st.metric("💬 Messages", len(result.get('conversation', [])))
                        with col3:
                            st.metric("🔧 Steps", len(result.get('step_history', [])))
                        with col4:
                            st.metric("📊 Success", "✅" if result['answer'] != "No final answer provided" else "❌")

                        if auto_save:
                            result_json = json.dumps(result, indent=2, ensure_ascii=False)
                            timestamp = int(time.time())
                            filename = f"video_qa_result_{timestamp}.json"
                            st.download_button(
                                label="📁 Download Results",
                                data=result_json,
                                file_name=filename,
                                mime="application/json",
                                use_container_width=True
                            )
                except Exception as e:
                    st.error(f"❌ Processing failed: {e}")
                    if enable_debug:
                        st.exception(e)
            else:
                st.error("❌ Video file not found!")
        elif submitted:
            st.error("❌ Please fill in both question and video path!")

    with tab2:
        st.info("🔒 Examples are now shown directly above the input. This tab is deprecated.")


    with tab3:
        st.markdown("### ⚙️ Advanced Settings")
        st.info("Advanced configuration options will be available here")


class StreamlitVideoQAManager:
    """Streamlit兼容的视频问答管理器"""
    
    def __init__(self, args, step_callback: Optional[Callable] = None):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_duration = args.clip_duration
        self.use_subtitle = args.use_subtitle
        self.clip_fps = args.clip_fps
        self.step_callback = step_callback or self._default_step_callback
        
        # # API配置
        self.ds_model_name = os.getenv('API_MODEL_NAME')
        self.ds_api_base = os.getenv('API_BASE_URL')
        self.ds_api_keys = [os.getenv('API_KEY')]

        # VLM服务器配置
        self.vlm_model_name = args.mllm_path
        self.vlm_api_key = "EMPTY"
        self.vlm_api_base = f"http://0.0.0.0:12345/v1"
        
        # 初始化组件
        self._initialize_components()
        
        # 对话状态
        self.messages = []
        self.cur_turn = 0
        self.current_data = {}
        self.step_history = []
        
    def _default_step_callback(self, step_data):
        """默认步骤回调"""
        self.step_history.append(step_data)
        
    def _initialize_components(self):
        """初始化各个组件"""
        try:
            # 初始化检索器
            self.clip_save_folder = f'{self.args.dataset_folder}/clips/{self.args.clip_duration}/'
            self.args.retriever_type = 'large'
            self.retriever = Retrieval_Manager(args=self.args, clip_save_folder=self.clip_save_folder)
            self.retriever.load_model_to_gpu(0)
            st.success("✅ Retrieval Manager initialized")
            
        except Exception as e:
            st.error(f"❌ Failed to initialize components: {e}")
            raise
    
    def preprocess_video_with_progress(self, video_path):
        """带进度显示的视频预处理"""
        video_name = video_path.split('/')[-1][:-4]
        
        # 创建输出目录
        clips_dir = os.path.join(self.args.dataset_folder, 'clips', '10', video_name)
        frames_dir = os.path.join(self.args.dataset_folder, 'dense_frames', video_name)
        os.makedirs(clips_dir, exist_ok=True)
        os.makedirs(frames_dir, exist_ok=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 获取视频时长
            status_text.text("📊 Analyzing video...")
            duration = _get_video_duration(video_path)
            progress_bar.progress(0.2)
            
            # 并行处理视频切分和帧提取
            status_text.text("✂️ Cutting video clips...")
            with ThreadPoolExecutor(max_workers=2) as executor:
                clip_future = executor.submit(_cut_video_clips, video_path, clips_dir, video_name, duration)
                progress_bar.progress(0.5)
                
                status_text.text("🎞️ Extracting frames...")  
                frame_future = executor.submit( video_path, frames_dir)
                progress_bar.progress(0.8)
                
                clip_future.result()
                frame_future.result()
                progress_bar.progress(1.0)
            
            status_text.text("✅ Video preprocessing completed!")
            
            self.step_callback({
                'type': 'preprocessing',
                'status': 'completed',
                'video_name': video_name,
                'duration': duration
            })
            
        except Exception as e:
            st.error(f"❌ Video preprocessing failed: {e}")
            raise
    
    def single_text2text_with_callback(self, message):
        """带回调的文本生成"""
        self.step_callback({
            'type': 'llm_call',
            'status': 'started',
            'input_length': len(str(message))
        })
        
        # st.markdown("**Calling DeepSeek...**")
        with st.spinner("Calling DeepSeek..."):
            llm = OpenAI(base_url=self.ds_api_base, api_key=self.ds_api_keys[0])
            
            for retry in range(3):
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(
                            lambda: llm.chat.completions.create(
                                model=self.ds_model_name,
                                messages=message,
                            )
                        )
                        completion = future.result(timeout=1800)
                        response = completion.choices[0].message.content.strip()
                        
                        self.step_callback({
                            'type': 'llm_call',
                            'status': 'completed',
                            'response_length': len(response),
                            'retry_count': retry
                        })
                        
                        return response
                        
                except Exception as e:
                    self.step_callback({
                        'type': 'llm_call',
                        'status': 'error',
                        'error': str(e),
                        'retry_count': retry
                    })
                    if retry < 2:
                        time.sleep(5)
                    else:
                        return ""
        
    def process_tools_with_callback(self, output_text, video_path, duration):
        """带回调的工具处理"""
        tool_result = ''
        tools_used = []
        
        # 视频阅读器
        if "<video_reader_question>" in output_text:
            tools_used.append('video_reader')
            self.step_callback({
                'type': 'tool_call',
                'tool': 'video_reader',
                'status': 'started'
            })
            
            pattern = r"<video_reader>([^<]+)</video_reader>\s*<video_reader_question>([^<]+)</video_reader_question>"
            output_text_clean = output_text.split('</thinking>')[1] if '</thinking>' in output_text else output_text
            matches = re.findall(pattern, output_text_clean)
            
            queries, video_paths = [], []
            for match in matches:
                match_set, query = match[0], match[1]
                
                if ':' in match_set:
                    video_clip = []
                    for m in match_set.split(';'):
                        if ':' in m:
                            begin_time_stamp, end_time_stamp = m.split(':')
                            begin_time_stamp, end_time_stamp = float(begin_time_stamp), float(end_time_stamp)
                        else:
                            begin_time_stamp = end_time_stamp = float(m)
                        cur_video_clip,_ = timestamp_to_clip_path(
                            self.args.dataset_folder, begin_time_stamp, end_time_stamp, 
                            video_path, fps=self.args.clip_fps
                        )
                        video_clip += cur_video_clip
                else:
                    clip_numbers = sorted([int(m) for m in match_set.split(';') if m.isdigit()])
                    video_clip,_ = clip_number_to_clip_path(
                        self.args.dataset_folder, clip_numbers, video_path, 
                        clip_duration=self.clip_duration, fps=self.clip_fps
                    )
                
                query = (
                    "Please watch the given video and answer the following question: " + query +
                    " Provide your answer along with a detailed description of the video."
                )
                queries.append(query)
                video_paths.append(video_clip)
            
            ans_li = self.batch_video2text_server(queries, video_paths)
            for match, ans in zip(matches, ans_li):
                match_set, query = match[0], match[1]
                tool_result += f'Video reader result for {match_set}: {ans}\n'
            
            self.step_callback({
                'type': 'tool_call',
                'tool': 'video_reader',
                'status': 'completed',
                'queries_count': len(queries),
                'result_length': len(tool_result)
            })
        
        # 文本检索器
        if '<video_segment_retriever_textual_query>' in output_text:
            tools_used.append('text_retriever')
            self.step_callback({
                'type': 'tool_call',
                'tool': 'text_retriever',
                'status': 'started'
            })

            pattern = r"<video_segment_retriever_textual_query>(.*?)</video_segment_retriever_textual_query>"
            if '</thinking>' in output_text:
                output_text_clean = output_text.split('</thinking>')[1]
            else:
                output_text_clean = output_text
                
            matches = re.findall(pattern, output_text_clean, flags=re.DOTALL)
            topk_pattern = r"<topk>(.*?)</topk>"
            topk_matches = re.findall(topk_pattern, output_text_clean, flags=re.DOTALL)
            
            for j, match_set in enumerate(matches):
                for match in match_set.split(';'):
                    try:
                        topk = int(topk_matches[j]) if j < len(topk_matches) else 5
                    except:
                        topk = 5
                    
                    video_clip_paths = self.retriever.get_informative_clips(
                        match, video_path=video_path, top_k=topk, total_duration=duration
                    )
                    cur_video_paths = [int(video[0].split('/')[-1].split('_')[1]) for video in video_clip_paths]
                    tool_result += f"The tool results for <video_segment_retriever_textual_query>{match}</video_segment_retriever_textual_query> are:\n" + str(cur_video_paths) + '\n'

             
            self.step_callback({
                'type': 'tool_call',
                'tool': 'text_retriever',
                'status': 'completed'
            })


        # 图像检索器
        if '<video_segment_retriever_image_query>' in output_text:
            tools_used.append('image_retriever')
            self.step_callback({
                'type': 'tool_call',
                'tool': 'image_retriever', 
                'status': 'started'
            })
            
            pattern = r"<video_segment_retriever_image_query>(.*?)</video_segment_retriever_image_query>"
            if '</thinking>' in output_text:
                output_text_clean = output_text.split('</thinking>')[1]
            else:
                output_text_clean = output_text
                
            matches = re.findall(pattern, output_text_clean, flags=re.DOTALL)
            topk_pattern = r"<topk>(.*?)</topk>"
            topk_matches = re.findall(topk_pattern, output_text_clean, flags=re.DOTALL)
            
            for j, match_set in enumerate(matches):
                for match in match_set.split(';'):
                    try:
                        topk = int(topk_matches[j]) + 1 if j < len(topk_matches) else 10
                    except:
                        topk = 10
                    
                    begin, end = float(match) - 1, float(match) + 1
                    query_video_path = extract_video_clip(video_path, begin, end)
                    video_clip_paths = self.retriever.get_informative_clips_with_video_query(
                        self.current_data.get("question", ""), query_video_path, 
                        video_path=video_path, top_k=topk, total_duration=duration
                    )
                    
                    cur_video_paths = []
                    for video in video_clip_paths:
                        clip_number = int(video[0].split('/')[-1].split('_')[1])
                        if not clip_number * self.clip_duration <= float(match) <= clip_number * self.clip_duration + self.clip_duration:
                            cur_video_paths.append(clip_number)
                    
                    tool_result += f"The tool results for <video_segment_retriever_image_query>{match}</video_segment_retriever_image_query> are:\n" + str(cur_video_paths) + '\n'

            self.step_callback({
                'type': 'tool_call',
                'tool': 'image_retriever',
                'status': 'completed'
            })
        
            
        # 字幕相关工具
        if '<subtitle_retriever>' in output_text:
            tools_used.append('subtitle_retriever')
            self.step_callback({
                'type': 'tool_call',
                'tool': 'subtitle_retriever', 
                'status': 'started'
            })
            pattern = r"<subtitle_retriever>(.*?)</subtitle_retriever>"
            if '</thinking>' in output_text:
                output_text_clean = output_text.split('</thinking>')[1]
            else:
                output_text_clean = output_text
                
            matches = re.findall(pattern, output_text_clean, flags=re.DOTALL)
            topk_pattern = r"<topk>(.*?)</topk>"
            topk_matches = re.findall(topk_pattern, output_text_clean, flags=re.DOTALL)

            for j, match_set in enumerate(matches):
                subtitle_triples = []
                vis = []
                for match in match_set.split(';'):
                    topk = int(topk_matches[j]) if len(topk_matches) > j else 30
                    cur_subtitle_triples = self.retriever.get_informative_subtitles(
                        match, video_path=video_path, top_k=topk, total_duration=duration
                    )
                    
                    for x in cur_subtitle_triples:
                        if x[0] not in vis:
                            begin_timestamp, end_timestamp = x[0], x[1]
                            subtitle_triples.append({
                                'begin_timestamp': begin_timestamp, 
                                'end_timestamp': end_timestamp, 
                                'text': x[2]
                            })
                            vis.append(x[0])
                
                subtitle_triples = sorted(subtitle_triples, key=lambda x: x['begin_timestamp'])
                tool_result += f"The tool results for <subtitle_retriever>{match_set}</subtitle_retriever> are:\n" + str(subtitle_triples) + '\n'

            self.step_callback({
                'type': 'tool_call',
                'tool': 'subtitle_retriever', 
                'status': 'completed'
            })  
        if '<subtitle_extractor>' in output_text:
            tools_used.append('subtitle_extractor')
            self.step_callback({
                'type': 'tool_call',
                'tool': 'subtitle_extractor', 
                'status': 'started'
            })  

            pattern = r"<subtitle_extractor>(.*?)</subtitle_extractor>"
            if '</thinking>' in output_text:
                output_text_clean = output_text.split('</thinking>')[1]
            else:
                output_text_clean = output_text
                
            matches = re.findall(pattern, output_text_clean, flags=re.DOTALL)

            for match_set in matches:
                for match in match_set.split(';'):
                    begin_timestamp, end_timestamp = float(match.split(':')[0]), float(match.split(':')[1])
                    all_subtitle_triples = extract_subtitles(video_path)
                    cur_subtitle_triples = [
                        {'begin_timestamp': int(x[0]), 'end_timestamp': int(x[1]), 'subtitle': x[2]} 
                        for x in all_subtitle_triples if begin_timestamp <= x[0] <= end_timestamp
                    ]
                    tool_result += f"The tool results for <subtitle_extractor>{match}</subtitle_extractor> are:\n" + str(cur_subtitle_triples) + '\n'
        
        
            self.step_callback({
                'type': 'tool_call',
                'tool': 'subtitle_extractor', 
                'status': 'completed'
            })  

        self.step_callback({
            'type': 'tools_summary',
            'tools_used': tools_used,
            'total_result_length': len(tool_result)
        })
        
        return tool_result
    
    def batch_video2text_server(self, queries, video_paths):
        """批量视频到文本处理"""
        messages = []
        for query, video_path in zip(queries, video_paths):
            base64Frames = image_paths_to_base64(video_path)
            cur_message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{x}", "detail": "low"}}, base64Frames),
                    ],
                },
            ]
            messages.append(cur_message)
        
        results = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(self.single_video2text, msg) for msg in messages]
            
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 更新进度
                    progress = (i + 1) / len(futures)
                    if hasattr(st, 'session_state') and 'vlm_progress' in st.session_state:
                        st.session_state.vlm_progress.progress(progress)
                        
                except Exception as e:
                    st.error(f"VLM processing error: {e}")
                    results.append("")
        
        return results
    
    def single_video2text(self, message):
        """单次视频到文本生成"""
        llm = OpenAI(base_url=self.vlm_api_base, api_key=self.vlm_api_key)
        
        for retry in range(5):
            try:
                completion = llm.chat.completions.create(
                    model=self.vlm_model_name,
                    messages=message,
                    max_tokens=8192,
                )
                return completion.choices[0].message.content.strip()
            except Exception as e:
                if retry < 4:
                    time.sleep(10)
                else:
                    st.error(f"VLM API error: {e}")
                    return ""
    
    def process_single_input_with_ui(self, question: str, video_path: str, subtitle_path: str = None) -> Dict:
        """带UI更新的单个输入处理"""
        
        # 创建UI容器
        main_container = st.container()
        progress_container = st.container()
        steps_container = st.container()
        
        with main_container:
            st.markdown("### 🚀 Processing Started")
            
        with progress_container:
            overall_progress = st.progress(0)
            current_step_text = st.empty()
            
        # 重置状态
        self.messages = []
        self.cur_turn = 0
        self.step_history = []
        
        # 预处理视频
        current_step_text.text("📹 Preprocessing video...")
        self.preprocess_video_with_progress(video_path)
        overall_progress.progress(0.1)
        
        # 获取视频信息
        with VideoFileClip(video_path) as clip:
            duration = clip.duration
        

        subtitles = self.get_subtitles(subtitle_path, video_path)
        # 构建初始数据
        self.current_data = {
            'question': question,
            'video_path': video_path,
            'duration': duration,
            'subtitles': subtitles
        }
        
        # 构建初始提示
        initial_prompt = self.build_initial_prompt(question, duration, subtitles)
        
        # 初始化对话
        self.messages = [{
            "role": "user",
            "content": initial_prompt
        }]
        
        current_step_text.text("🤖 Starting conversation...")
        overall_progress.progress(0.2)
        
        # 对话循环
        MAX_DS_ROUND = 20

        # Then start the conversation loop
        while self.cur_turn < MAX_DS_ROUND:
            step_progress = 0.2 + (0.7 * (self.cur_turn + 1) / MAX_DS_ROUND)
            overall_progress.progress(step_progress)
            current_step_text.text(f"💬 Processing round {self.cur_turn + 1}/{MAX_DS_ROUND}")
                            
            # 在steps容器中显示当前步骤
            with steps_container:
                if self.cur_turn==0:
                    # Display system prompt at the start
                    st.markdown("**📋 System Prompt:**")         
                    last_user_msg = next((msg['content'] for msg in reversed(self.messages) if msg['role'] == 'user'), "")         
                    st.text_area("", last_user_msg, height=200, key=f"user_input_{self.cur_turn}")


                with st.expander(f"🔄 Round {self.cur_turn + 1}", expanded=True):
                    # 获取模型回复
                    response = self.single_text2text_with_callback(self.messages)
                    if not response:
                        st.error("❌ Failed to get model response")
                        break
                                        
                    self.messages.append({"role": "assistant", "content": response})
                                        
                    # 助手回复部分
                    st.markdown("**🧠 Reasoning:**")
                    st.text_area("", response, height=200, key=f"assistant_response_{self.cur_turn}")
                                        
                    # 检查是否有最终答案
                    if '<answer>' in response:
                        final_answer = self.extract_final_answer(response)
                        st.success(f"🎯 **Final Answer Found:** {final_answer}")
                                                
                        overall_progress.progress(1.0)
                        current_step_text.text("✅ Processing completed!")
                                                
                        return {
                            "question": question,
                            "answer": final_answer,
                            'video_path': video_path,
                            "conversation": self.messages,
                            "rounds": self.cur_turn + 1,
                            "step_history": self.step_history
                        }
                                        
                    # 处理工具调用
                    # st.markdown("**Tool Processing...**")
                    with st.spinner("Processing tools..."):
                        tool_result = self.process_tools_with_callback(response, video_path, duration)
                                        
                    if tool_result:
                        st.markdown("**📊 Tool Results:**")
                        st.code(tool_result, language="text")
                                                
                        # 添加工具结果到对话
                        remaining_rounds = MAX_DS_ROUND - self.cur_turn - 1
                        tool_message = tool_result + f"\nYou have {remaining_rounds} rounds remaining. Please make the most of each opportunity!"
                        self.messages.append({"role": "user", "content": tool_message})
                    else:
                        # 处理无效输出
                        if self.cur_turn >= MAX_DS_ROUND - 1:
                            self.messages.append({
                                "role": "user",
                                "content": "Maximum rounds reached! Please provide your final answer in <answer></answer> format!"
                            })
                            st.warning("⚠️ Maximum rounds reached, requesting final answer...")
                        else:
                            self.messages.append({
                                "role": "user",
                                "content": "Invalid output format! Please use the provided XML format for tools or provide final answer in <answer></answer>."
                            })
                            st.warning("⚠️ Invalid format detected, requesting correction...")
                        
            self.cur_turn += 1
            time.sleep(0.5)  # 小延迟让用户看到进度

        # 如果达到最大轮数仍未得到答案
        overall_progress.progress(1.0)
        current_step_text.text("⚠️ Maximum rounds reached without final answer")

        return {
            "question": question,
            "answer": "No final answer provided within maximum rounds",
            'video_path': video_path,
            "conversation": self.messages,
            "rounds": self.cur_turn,
            "step_history": self.step_history
        }
    
    def get_subtitles(self, subtitle_path, video_path):
        """获取字幕信息"""
        if not subtitle_path or not os.path.exists(subtitle_path):
            return ""
            
        subtitles = ""
        try:
            if subtitle_path.endswith('.srt'):
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
                                pattern = r'<font color="white" size=".72c">(.*?)</font>'
                                raw_text = re.findall(pattern, text, flags=re.DOTALL)
                                try:
                                    text = raw_text[0]
                                except:
                                    text = text

                                subtitles += str(int(start_time)) + '-' + str(int(end_time)) +':'+ text + ' '
                                
            elif subtitle_path.endswith('.json'):
                subtitles_data = json.load(open(subtitle_path))
                subtitles_list = [str(int(parse_subtitle_time(dic['start']))) + '-' +str(int(parse_subtitle_time(dic['end']))) +':'+ dic['line'] for dic in subtitles_data]
                subtitles = '\n'.join(subtitles_list)
                
        except Exception as e:
            st.warning(f"Failed to load subtitles: {e}")
            
        return subtitles

    def build_initial_prompt(self, question, duration,subtitles):
        """构建初始提示模板"""
        if subtitles:
            prompt = initial_input_template_general_r1_subtitle.format(
                question=question,
                duration=duration,
                clip_duration=self.clip_duration,
                MAX_DS_ROUND=20
            )
        else:
            prompt = initial_input_template_general_r1.format(
                question=question,
                duration=duration,
                clip_duration=self.clip_duration,
                MAX_DS_ROUND=20
            )
        
        return prompt

    def extract_final_answer(self, text: str) -> str:
        """提取最终答案"""
        try:
            answer_content = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)[-1].strip()
            # 查找第一个大写字母
            first_upper = re.search(r'[A-Z]', answer_content)
            return first_upper.group(0) if first_upper else answer_content.strip()
        except:
            return text.strip()


if __name__ == "__main__":
    create_enhanced_demo()