import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import os
os.environ["VLLM_USE_MODELSCOPE"] = "false"   
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
import glob
from video_utils  import _get_video_duration,_cut_video_clips,extract_subtitles,timestamp_to_clip_path,is_valid_video,is_valid_frame,extract_video_clip,parse_subtitle_time,clip_number_to_clip_path,image_paths_to_base64,load_image,robust_eval, timestamp_to_frames
import json
import re
import torch
import decord
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
from decord import VideoReader, cpu
from PIL import Image
from pathlib import Path
import argparse
from retriever_languagebind import Retrieval_Manager
from prompt import *
from collections import defaultdict
import random
from qwen_vl_utils import process_vision_info
import time
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor 
from vllm import LLM, EngineArgs, SamplingParams
from openai import OpenAI
from multiprocessing import Pool, cpu_count, Manager
from concurrent.futures import ThreadPoolExecutor, as_completed
from moviepy.video.io.VideoFileClip import VideoFileClip
import pickle
from vllm import LLM, EngineArgs, SamplingParams
import os
from PIL import Image
import io
from multiprocessing import Pool, cpu_count
from functools import partial

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



MAX_DS_ROUND = 20
os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.backends.cuda.matmul.allow_tf32 = True


def is_list(s):
    pattern = r'\((\d+),\s*(\d+),\s*"([^"]*)"\)'
    match = re.match(pattern, s)
    if match:
        return True
    else:
        return False

class SingleSampleProcessor:
    
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temporal_grounding_model_name = os.getenv('API_MODEL_NAME_TEMPORAL_GROUNDING', 'deepseek-v3-250324')
        self.temporal_grounding_api_base = os.getenv('API_BASE_URL_TEMPORAL_GROUNDING', 'https://api.deepseek.com/v1').split(',')
        self.temporal_grounding_api_keys = os.getenv('API_KEY_TEMPORAL_GROUNDING').split(',')
        
        self.ds_model_name = os.getenv('API_MODEL_NAME')
        self.ds_api_base = os.getenv('API_BASE_URL').split(',')
        self.ds_api_keys = os.getenv('API_KEY').split(',')

        self.vlm_model_name = os.getenv('API_MODEL_NAME_VLM')

        print("Initializing VLLM server with conservative settings...")
        self.vlm_server = LLM(
            model = os.getenv('API_MODEL_NAME_VLM'), 
            gpu_memory_utilization=0.93,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=50000,
            enable_chunked_prefill=True,
            enforce_eager=True,
        )

        self.processor = AutoProcessor.from_pretrained(os.getenv('API_MODEL_NAME_VLM'), use_fast=True)
        self.processor.tokenizer.padding_side = 'left'

        self.args.dataset_folder = f'{self.args.dataset_folder}/{args.dataset}'
        self.clip_save_folder = f'{self.args.dataset_folder}/clips/{args.clip_duration}/'
        self.retriever = Retrieval_Manager(args, clip_save_folder=self.clip_save_folder)
        
        gpu_id = args.thread_idx % 8 if len(os.getenv('CUDA_VISIBLE_DEVICES', '0').split(',')) == 8 else 0

        self.retriever.load_model_to_gpu(gpu_id)
        self.experience_pool = []


    def get_dic_subtitles(self, dic):
        
        video_id = dic['video_path'].split('/')[-1].split('.')[0]

        if os.path.exists(f'{self.args.dataset_folder}/{self.args.dataset}/subtitles/{video_id}.srt'):
            subtitle_path = f'{self.args.dataset_folder}/{self.args.dataset}/subtitles/{video_id}.srt'
            subtitles = ''
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

                            subtitles += str(int(start_time)) + '-' + str(int(end_time)) + ':' + text + ' '
            dic['subtitles'] = subtitles
        elif os.path.exists(f'{self.args.dataset_folder}/{self.args.dataset}/subtitles/{video_id}.json'):
            subtitle_path = f'{self.args.dataset_folder}/{self.args.dataset}/subtitles/{video_id}.json'
            subtitles = json.load(open(subtitle_path))
            subtitles = [str(int(parse_subtitle_time(sub['start']))) + '-' + str(int(parse_subtitle_time(sub['end']))) + ':' + sub['line'] for sub in subtitles]
            dic['subtitles'] = '\n'.join(subtitles)
        else:
            dic['subtitles'] = ''
        return dic

    def build_initial_prompt(self, data: Dict) -> str:
        

        data['question'] = data['question'].replace('Only give the best option. Best option: (','')
        data['options'] = [] if 'options' not in data else data['options']
        if self.args.use_subtitle:
            base_prompt = initial_input_template_subtitle.format(
                        question=data['question'] + "\n" + "\n".join(data['options']),
                        duration=data['duration'],
                        clip_duration=self.args.clip_duration,
                        MAX_DS_ROUND=MAX_DS_ROUND
                    )
        else:
            base_prompt = initial_input_template_wo_subtitle.format(
                        question=data['question'] + "\n" + "\n".join(data['options']),
                        duration=data['duration'],
                        clip_duration=self.args.clip_duration,
                        MAX_DS_ROUND=MAX_DS_ROUND
                    )
        base_prompt = base_prompt.replace('thinking>', 'think>')
        return base_prompt

    
    def single_text2text(self, message, model_name, base_url=None, api_keys=None) -> str:
        
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
                    print(f'[TEXT2TEXT] cost time:', time.time()-start_time)
                    return ans
                except Exception as e:
                    print('[TEXT2TEXT] ERROR:', e)
                    safe_write_with_lock({'model': model_name, 'input': message},read_file)
                    os.system(f'rm {output_file}')

            if time.time()-start_time>120:
                break
            if not os.path.exists(read_file):
                safe_write_with_lock({'model': model_name, 'input': message},read_file)
            # print('waiting for', output_file)
            time.sleep(0.2)

        print('[TEXT2TEXT] ERROR: Timeout, model:', model_name)
        return ''
    
    def single_text2text_vllm(self, message, model_name, base_url=None, api_keys=None) -> str:
        start_time = time.time()
        llm = OpenAI(base_url=random.choice(base_url), api_key=random.choice(api_keys))
        retry = 0
        while retry < 3:
            try:
                completion = llm.chat.completions.create(
                    model=model_name,
                    messages=message,
                )
                ans = completion.choices[0].message.content.strip('\n').strip()
                return ans
            except Exception as e:
                print(f"Text2Text API error: {e}")
                retry += 1
                time.sleep(10)
        return ""
    
    
    def batch_video2text(self, task_li, batch_size=4):

        start_time = time.time()

        prompt_li, image_paths_li, timestamps_li = [l[0] for l in task_li], [l[1] for l in task_li], [l[2] for l in task_li]
        ans_li = []

        for batch_idx in range(0, len(task_li), batch_size):
            batch_end = min(batch_idx + batch_size, len(task_li))
            batch_prompts = prompt_li[batch_idx:batch_end]
            batch_image_paths = image_paths_li[batch_idx:batch_end]
            batch_timestamps = timestamps_li[batch_idx:batch_end]
            batch_results = self._process_batch(batch_prompts, batch_image_paths, batch_timestamps)
            ans_li.extend(batch_results)

        total_time = time.time() - start_time
        print(f'VLM batch processing completed: {total_time:.2f}s, Tasks: {len(task_li)}, avg per task: {total_time/len(task_li):.2f}s')
        
        return ans_li
    
    def _process_batch(self, prompts, image_paths_li, timestamps_li):
        batch_inputs = []
        
        for prompt, image_paths, timestamps in zip(prompts, image_paths_li, timestamps_li):
            content = []
            image_data = []
            for idx, image_path in enumerate(image_paths):
                if os.path.exists(image_path):
                    try: 
                        image = Image.open(image_path)
                        image.verify()
                        image = Image.open(image_path)
                        
                        width, height = image.size
                        if max(width, height) > 768:
                            if width > height:
                                new_width = 768
                                new_height = int(height * (768 / width))
                            else:
                                new_height = 768
                                new_width = int(width * (768 / height))
                            target_size = (new_width, new_height)
                        else:
                            target_size = (width, height)
                        
                        if image.size != target_size:
                            image = image.resize(target_size, Image.Resampling.LANCZOS)
                        
                        image_data.append(image)
                        
                    except Exception as e:
                        print(f"Invalid image file: {image_path} - {str(e)}")
                        continue
                        

            content.append({
                "type": "video",
                "video": image_paths,
            })
            
            content.append({
                "type": "text",
                "text": prompt,
            })

            messages = [{
                "role": "user",
                "content": content,
            }]

            formatted_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            fps = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 2.0

            batch_input = {
                "prompt": formatted_prompt,
                "multi_modal_data": {"video": image_data},
                "mm_processor_kwargs": {
                    "min_pixels": 4 * 28 * 28,
                    "max_pixels": 768 * 28 * 28,
                    "fps": fps,
                },
            }
            batch_inputs.append(batch_input)
            

        valid_inputs = [inp for inp in batch_inputs if inp is not None]
        valid_indices = [i for i, inp in enumerate(batch_inputs) if inp is not None]
        
        if not valid_inputs:
            print("Warning: No valid inputs in batch")
            return [""] * len(prompts)

        sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
        
        batch_start_time = time.time()
        outputs = self.vlm_server.generate(
            valid_inputs,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        batch_time = time.time() - batch_start_time
        
        results = [""] * len(prompts)
        for i, output in enumerate(outputs):
            if i < len(valid_indices):
                original_idx = valid_indices[i]
                result_text = output.outputs[0].text.strip()
                results[original_idx] = self.remove_duplicate_sentences(result_text)
                
        return results
        
    
    def _fallback_individual_processing(self, batch_prompts, batch_image_paths, batch_timestamps):
        
        print("Falling back to individual processing...")
        results = []
        
        for prompt, image_paths, timestamps in zip(batch_prompts, batch_image_paths, batch_timestamps):
            try:
                content = []
                image_data = []

                for idx, image_path in enumerate(image_paths):
                    if os.path.exists(image_path):
                        try: 
                            image = Image.open(image_path)
                            image.verify()
                            image = Image.open(image_path)
                            
                            width, height = image.size
                            if max(width, height) > 768:
                                if width > height:
                                    new_width = 768
                                    new_height = int(height * (768 / width))
                                else:
                                    new_height = 768
                                    new_width = int(width * (768 / height))
                                target_size = (new_width, new_height)
                            else:
                                target_size = (width, height)
                            
                            if image.size != target_size:
                                image = image.resize(target_size, Image.Resampling.LANCZOS)
                            
                            image_data.append(image)
                            
                        except Exception as e:
                            print(f"Invalid image file: {image_path} - {str(e)}")
                            continue



                content.append({
                    "type": "video", 
                    "video": image_paths,
                })
                
                content.append({
                    "type": "text",
                    "text": prompt,
                })

                messages = [{
                    "role": "user",
                    "content": content,
                }]

                formatted_prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
                fps = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 2.0
                

                outputs = self.vlm_server.generate(
                    {
                        "prompt": formatted_prompt,
                        "multi_modal_data": {"video": image_data},
                        "mm_processor_kwargs": {
                            "min_pixels": 28 * 28,
                            "max_pixels": 768 * 28 * 28,
                            "fps": fps,
                        },
                    },
                    sampling_params=sampling_params,
                    use_tqdm=False
                )
                
                result_text = outputs[0].outputs[0].text.strip()
                results.append(self.remove_duplicate_sentences(result_text))
                
            except Exception as e:
                print(f"Error in individual processing: {e}")
                results.append("")
        
        return results


    def extract_final_answer(self, text: str) -> str:
        
        try:
            answer_content = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)[-1].strip()
            first_upper = re.search(r'[A-Z]', answer_content)
            return first_upper.group(0) if first_upper else '-'
        except:
            return '-'

    def process_single_sample(self, dic: Dict) -> Dict:
        
        torch.cuda.empty_cache()
        if self.args.use_subtitle:
            dic = self.get_dic_subtitles(dic)
        
        messages = []
        initial_prompt = self.build_initial_prompt(dic)
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": initial_prompt}]
        })
        
        cur_turn = 0
        
        while cur_turn < MAX_DS_ROUND:
            output_text = self.single_text2text(messages, self.ds_model_name, self.ds_api_base, self.ds_api_keys)
            
            if not output_text:
                break
                
            messages.append({'role': 'assistant', 'content': output_text})
            cur_turn += 1
            
            if '<answer>' in output_text:
                answer = self.extract_final_answer(output_text)
                dic['pred_answer'] = answer
                score = dic['pred_answer'] == dic['answer'][0]
                if 'subtitles' in dic:
                    del dic['subtitles']
                    
                return {
                    'messages': messages,
                    'raw_data': dic,
                    'score': score
                }
            
            tool_result = self.process_tool_calls(output_text, dic)
            
            if tool_result:
                messages.append({
                    'role': 'user',
                    'content': tool_result + f"\nYou have now engaged in a total of {cur_turn} rounds of conversation, with {MAX_DS_ROUND-cur_turn} calls remaining. Please make the most of each opportunity until you obtain an accurate answer. Don't guess the answer!! Obtain an accurate answer with tools!!"
                })
            elif '<answer>' not in output_text:
                messages.append({
                    'role': 'user',
                    'content': 'The output is invalid. You should strictly follow the provided xml format!!!'
                })
            
            if cur_turn >= MAX_DS_ROUND:
                messages.append({
                    'role': 'user',
                    'content': 'Maximum number of rounds reached! Now you should output the final answer within <answer></answer>!!!'
                })
        
        if 'subtitles' in dic:
            del dic['subtitles']
        
        return {
            'messages': messages,
            'raw_data': dic,
            'score': 0
        }

    def process_tool_calls(self, output_text: str, dic: Dict) -> str:
        
        tool_result = ''
        video_path = dic['video_path']
        duration = dic['duration']
        
        if "<temporal_grounding_agent>" in output_text:
            valid, retry = 0, 0
            while not valid and retry<3:
                cur_tool_result, dic = self.process_temporal_grounding(output_text, duration, dic)
                valid = 1
                
            tool_result += cur_tool_result

        if "<video_reader_question>" in output_text:
            tool_result += self.process_video_reader(output_text, video_path)
    
        if '<video_segment_retriever_textual_query>' in output_text:
            tool_result += self.process_video_segment_retriever_text(output_text, video_path, duration)

        if '<video_segment_retriever_image_query>' in output_text:
            tool_result += self.process_video_segment_retriever_image(output_text, video_path, duration, dic)

        if '<subtitle_retriever>' in output_text:
            tool_result += self.process_subtitle_retriever(output_text, video_path, duration, dic)
        
        if '<subtitle_extractor>' in output_text:
            tool_result += self.process_subtitle_extractor(output_text, video_path, dic)
        
        if "<video_browser>" in output_text:
            tool_result += self.process_video_browser(output_text, video_path, duration)
        
        return tool_result

    def process_temporal_grounding(self, output_text: str, duration: str, dic: Dict) -> str:
        def convert_timestamps(output_text):
            pattern = r"<video_reader>([^<]+)</video_reader>\s*<video_reader_question>([^<]+)</video_reader_question>"
            matches = re.findall(pattern, output_text.strip())
            if not matches:
                return output_text

            time_matches = [match[0] for match in matches]
            question_matches = [match[1] for match in matches]
            converted_times = []
            for time_match in time_matches:
                begin_time_stamp, end_time_stamp = time_match.split(':')[0].replace(' ',''), time_match.split(':')[1].replace(' ','')
                begin_time_stamp, end_time_stamp = int(float(begin_time_stamp) / 10 * args.clip_duration), int(float(end_time_stamp) / 10 * args.clip_duration)
                converted_times.append(f'{begin_time_stamp}:{end_time_stamp}')
            
            converted_output_text = [f"<video_reader>{t}</video_reader><video_reader_question>{q}</video_reader_question>" for t,q in zip(converted_times, question_matches)]
            return '\n'.join(converted_output_text)

        pattern = r"<temporal_grounding_agent>([^<]+)</temporal_grounding_agent>"
        try:
            question = re.findall(pattern, output_text)[0]
        except:
            print("Warning: No valid temporal_grounding_agent matches found in the output text!!!", output_text)
            question = output_text.split('<temporal_grounding_agent>')[-1].split('</temporal_grounding_agent>')[0].strip()
        all_temporal_grounding_result = []
        if self.args.use_subtitle:
            agent_initial_prompt = initial_input_template_temporal_grounding_agent.format(clip_duration=10, question=question, duration=duration)
        else:
            agent_initial_prompt = initial_input_template_temporal_grounding_agent_wo_subtitle.format(clip_duration=10, question=question, duration=duration)
        
        agent_initial_prompt = agent_initial_prompt.replace('thinking>','think>')
        messages = [{
            "role": "user",
            "content": agent_initial_prompt
        }]
        
        cur_length = len(agent_initial_prompt.split())

        tool_call = self.single_text2text(messages, self.temporal_grounding_model_name, self.temporal_grounding_api_base, self.temporal_grounding_api_keys)
        
        
        for i in range(2):
            if '<video_reader>' in tool_call: 
                converted_tool_call = convert_timestamps(tool_call)   
                messages.append({
                    "role": "assistant",
                    "content": converted_tool_call
                })
                cur_length += len(converted_tool_call.split())
                tool_results = self.process_tool_calls(converted_tool_call, dic)
            else: 
                messages.append({
                    "role": "assistant",
                    "content": tool_call
                })
                cur_length += len(tool_call.split())
                tool_results = self.process_tool_calls(tool_call, dic)
                
                cur_length += len(tool_results.split())
                if cur_length > 32000:
                    print('The tool call exceeds the maximum length, truncate it.')
                    tool_results = tool_results[:32000 - len(agent_initial_prompt.split()) - 1000]
            
            
            cur_length += len(tool_results.split())
            if cur_length > 32000:
                print('The tool call exceeds the maximum length, truncate it.')
                tool_results = tool_results[:32000 - len(agent_initial_prompt.split()) - 2000]
        
            if i==0:
                tool_results += '\nNow you should call the video reader to check the video segments.'
                messages.append({
                    'role': 'user',
                    'content': tool_results
                })
                
                retry = 0
                while retry < 3:
                    tool_call = self.single_text2text(messages, self.temporal_grounding_model_name, self.temporal_grounding_api_base, self.temporal_grounding_api_keys)
                    if '<video_reader>' in tool_call and '[Pause]' in tool_call:
                        break
                    elif '<video_reader>' not in tool_call:
                        messages.append({
                            'role': 'assistant',
                            'content': tool_call
                        })
                        messages.append({
                            'role': 'user',
                            'content': 'You MUST call the video_reader to validate the retrieved segments! Follow the format: <video_reader>begin_timestamp:end_timestamp</video_reader><video_reader_question>your_question</video_reader_question> and then output [Pause]!'
                        })
                    retry += 1
                 
                
                if retry >= 3 and '<video_reader>' not in tool_call:
                    print('[TEMPORAL GROUNDING] Warning: Failed to get video_reader call after 3 retries')
                    tool_call = "No validation performed due to model non-compliance."
            
            else:
                tool_results += "\nNow you should output the final video segments."
                messages.append({
                    'role': 'user',
                    'content': tool_results
                })
                
                retry = 0
                while retry < 3:
                    tool_call = self.single_text2text(messages, self.temporal_grounding_model_name, self.temporal_grounding_api_base, self.temporal_grounding_api_keys)
                    
                    answer_matches = re.findall(r'<answer>([^<]+)</answer>', tool_call, re.DOTALL)
                    if answer_matches and '[pause]' not in answer_matches[-1].strip().lower():
                        break
                    elif not answer_matches:
                        messages.append({
                            'role': 'assistant', 
                            'content': tool_call
                        })
                        messages.append({
                            'role': 'user',
                            'content': 'You MUST output your final answer in the format: <answer>[(start, end, "description"), ...]</answer>. Do not include [Pause] in your answer!'
                        })
                    retry += 1
                    
                if retry >= 3:
                    print('[TEMPORAL GROUNDING] Warning: Failed to get valid answer after 3 retries')
                    tool_call = '<answer>[]</answer>' 

        if '<answer>' in tool_call: 
            tool_call = re.findall(r'<answer>([^<]+)</answer>', tool_call, re.DOTALL)[-1].strip()
            b_idx, e_idx = tool_call.find('['), tool_call.find(']')
            tool_call = tool_call[b_idx:e_idx+1]
            intervals = robust_eval(tool_call)
            all_temporal_grounding_result.extend(intervals)

        if 'search_history' not in dic:
            dic['search_history'] = []
        dic['search_history'].append(messages)
            
        all_temporal_grounding_result = f'There are {len(all_temporal_grounding_result)} related segments in the video: {all_temporal_grounding_result}'


        messages.append({
                'role': 'assistant',
                'content': all_temporal_grounding_result
            })

        return all_temporal_grounding_result, dic

    def process_video_reader(self, output_text: str, video_path: str) -> str:
        
        if '/thinking>' in output_text:
            output_text = output_text.split('/thinking>')[1]
        if '/think>' in output_text:
            output_text = output_text.split('/think>')[1]
    
        pattern = r"<video_reader>([^<]+)</video_reader>\s*<video_reader_question>([^<]+)</video_reader_question>"
        matches = re.findall(pattern, output_text.strip())

        if not matches:
            raise ValueError(f"Warning: No valid video_reader matches found in the output text: {output_text}")

        time_matches = [match[0] for match in matches]
        question_matches = [match[1] for match in matches]
        fps_matches = ['2.0'] * len(time_matches)  
        max_dimension_of_height_width_matches = ['768'] * len(time_matches)
    
        results = []

        error_result = ''
        
        tasks = []
        for query, time_match, fps, max_dimension_of_height_width in zip(question_matches, time_matches, fps_matches, max_dimension_of_height_width_matches):
            fps, max_dimension_of_height_width = float(fps), int(robust_eval(max_dimension_of_height_width))
            begin_time_stamp, end_time_stamp = time_match.split(':')[0], time_match.split(':')[1]
            begin_time_stamp, end_time_stamp = float(begin_time_stamp), float(end_time_stamp)

            if 'haystack' in  video_path:
                video_clip, timestamps = timestamp_to_frames(video_path, begin_time_stamp, end_time_stamp, f'{self.args.dataset_folder}/dense_frames')
                query_formatted = (
                    "Please analyze the provided video, the question is " + query + "\n Provide your response in this exact format:\n"
                    "The description of the video is: [Detailed description of visual content, actions, and context]\n"
                    "Textual Instructions on the screen: [Yes/No - if Yes, describe the text content]\n"
                    "The answer is: [Your answer to the question OR 'NONE' if textual instructions are present]\n(Only provide the answer to the question if NO textual instructions are present\n)"
                )
            else:
                video_clip, timestamps = timestamp_to_clip_path(self.args.dataset_folder, begin_time_stamp, end_time_stamp, video_path, fps=fps)

                query_formatted = (
                    "Please watch the given video and answer the following question: " + query +
                    "Output the detailed video description (including the object attribute, the relationship between object, the environment, etc), and the answer in this format: The description of the video is:YOUR_DESCRIPTION\nThe answer is:YOUR_ANSWER."
                )
            
            if len(video_clip)!=0:
                tasks.append((query_formatted, video_clip, timestamps))
            else:
                error_result += f"The video segment for <video_reader>{time_match}</video_reader> is empty. Please check the provided timestamps and ensure they are within the video's duration and the end_time_stamp is no less than the begin_time_stamp.\n"
                print(f"The video segment for <video_reader>{time_match}</video_reader> is empty. Please check the provided timestamps and ensure they are within the video's duration and the end_time_stamp is no less than the begin_time_stamp.")

        results = self.batch_video2text(tasks)

        ans = error_result
        for time_match, result in zip(time_matches, results):
            ans += f'The tool result for <video_reader>{time_match}</video_reader> is {result}\n'

        if not ans.strip():
            print("Warning: No valid results from video_reader processing!!!", output_text)
            return ""
        
        return ans

    def process_video_browser(self, output_text: str, video_path: str, duration: float) -> str:
        
        pattern = r"<video_browser>([^<]+)</video_browser>"
        if '/thinking>' in output_text:
            output_text = output_text.split('/thinking>')[1]
        if '/think>' in output_text:
            output_text = output_text.split('/think>')[1]
        queries = re.findall(pattern, output_text)
        
        if not queries:
            return ""
        
        query = queries[0]
        try:
            if 'haystack' in video_path:
                video_clip, timestamps = timestamp_to_frames(video_path,  0, duration, f'{self.args.dataset_folder}/dense_frames')
            else:
                video_clip, timestamps = timestamp_to_clip_path(self.args.dataset_folder, 0, duration, video_path, fps=self.args.clip_fps)
                
            ans = self.batch_video2text([(query, video_clip, timestamps)])[0]
            return f"The tool results for <video_browser>{query}</video_browser> is:{ans}\n"
        except Exception as e:
            print(f"Error processing video browser: {e}")
            return ""


    def process_video_segment_retriever_text(self, output_text: str, video_path: str, duration: float) -> str:
        
        pattern = r"<video_segment_retriever_textual_query>(.*?)</video_segment_retriever_textual_query>"
        if '/thinking>' in output_text:
            output_text = output_text.split('/thinking>')[1]
        if '/think>' in output_text:
            output_text = output_text.split('/think>')[1]
        matches = re.findall(pattern, output_text, flags=re.DOTALL)
        
        results = []
        for j, time_match in enumerate(matches):
            for match in time_match.split(';'):
                try:
                    topk = int(os.getenv('TOPK'))
                    video_clip_paths = self.retriever.get_informative_clips(match, video_path=video_path, top_k=topk, total_duration=duration)
                    cur_video_paths = [int(video[0].split('/')[-1].split('_')[1]) for video in video_clip_paths]
                    results.append(f"The tool results for <video_segment_retriever_textual_query>{match}</video_segment_retriever_textual_query> are:\n" + str(cur_video_paths) + '\n')
                except Exception as e:
                    print(f"Error processing video segment retriever text: {e}")
                    continue
        
        return ''.join(results)

    def process_video_segment_retriever_image(self, output_text: str, video_path: str, duration: float, dic: Dict) -> str:
        
        pattern = r"<video_segment_retriever_image_query>(.*?)</video_segment_retriever_image_query>"
        if '/thinking>' in output_text:
            output_text = output_text.split('/thinking>')[1]
        if '/think>' in output_text:
            output_text = output_text.split('/think>')[1]
        matches = re.findall(pattern, output_text, flags=re.DOTALL)
        
        pattern = r"<video_segment_retriever_image_query_text>(.*?)</video_segment_retriever_image_query_text>"
        matches_text = re.findall(pattern, output_text, flags=re.DOTALL)
        
        results = []
        for j, match, match_text in enumerate(zip(matches, matches_text)):  
            try:
                topk = int(os.getenv('TOPK'))
                begin, end = float(match) - 1, float(match) + 1
                query_video_path = extract_video_clip(video_path, begin, end)
                video_clip_paths = self.retriever.get_informative_clips_with_video_query(
                    match_text, query_video_path, 
                    video_path=video_path, top_k=topk, total_duration=duration
                )
                cur_video_paths = []
                for video in video_clip_paths:
                    clip_number = int(video[0].split('/')[-1].split('_')[1])
                    if not clip_number * self.args.clip_duration <= float(match) <= clip_number * self.args.clip_duration + self.args.clip_duration:
                        cur_video_paths.append(clip_number)
                results.append(f"The tool results for <video_segment_retriever_image_query>{match}</video_segment_retriever_image_query> are:\n" + str(cur_video_paths) + '\n')
            except Exception as e:
                print(f"Error processing video segment retriever image: {e}")
                continue
    
        return ''.join(results)

    def process_subtitle_retriever(self, output_text: str, video_path: str, duration: float, dic: Dict) -> str:
        
        pattern = r"<subtitle_retriever>(.*?)</subtitle_retriever>"
        if '/thinking>' in output_text:
            output_text = output_text.split('/thinking>')[1]
        if '/think>' in output_text:
            output_text = output_text.split('/think>')[1]
        matches = re.findall(pattern, output_text, flags=re.DOTALL)
        
        
        results = []
        for j, time_match in enumerate(matches):
            subtitle_triples = []
            vis = []
            for match in time_match.split(';'):
                try:
                    topk=10
                    cur_subtitle_triples = self.retriever.get_informative_subtitles(match, video_path=video_path, top_k=topk, total_duration=duration)
                    for x in cur_subtitle_triples:
                        if x[0] not in vis:
                            if 'starting_timestamp_for_subtitles' in dic:
                                begin_timestamp = x[0] - dic['starting_timestamp_for_subtitles']
                                end_timestamp = x[1] - dic['starting_timestamp_for_subtitles']
                            else:
                                begin_timestamp, end_timestamp = x[0], x[1]
                            subtitle_triples.append({'begin_timestamp': begin_timestamp, 'end_timestamp': end_timestamp, 'text': x[2]})
                            vis.append(x[0])
                except Exception as e:
                    print(f"Error processing subtitle retriever: {e}")
                    continue
            
            subtitle_triples = sorted(subtitle_triples, key=lambda x: x['begin_timestamp'])
            results.append(f"The tool results for <subtitle_retriever>{time_match}</subtitle_retriever> are:\n" + str(subtitle_triples) + '\n')
        
        return ''.join(results)

    def process_subtitle_extractor(self, output_text: str, video_path: str, dic: Dict) -> str:
        
        pattern = r"<subtitle_extractor>(.*?)</subtitle_extractor>"
        if '/thinking>' in output_text:
            output_text = output_text.split('/thinking>')[1]
        if '/think>' in output_text:
            output_text = output_text.split('/think>')[1]
        matches = re.findall(pattern, output_text, flags=re.DOTALL)
        
        results = []
        for time_match in matches:
            for match in time_match.split(';'):
                try:
                    begin_timestamp, end_timestamp = float(match.split(':')[0]), float(match.split(':')[1])
                    if 'starting_timestamp_for_subtitles' in dic:
                        begin_timestamp = begin_timestamp + dic['starting_timestamp_for_subtitles']
                        end_timestamp = end_timestamp + dic['starting_timestamp_for_subtitles']
                    
                    all_subtitle_triples = extract_subtitles(video_path)
                    cur_subtitle_triples = [
                        {'begin_timestamp': int(x[0]), 'end_timestamp': int(x[1]), 'subtitle': x[2]} 
                        for x in all_subtitle_triples if begin_timestamp <= x[0] <= end_timestamp
                    ]
                    results.append(f"The tool results for <subtitle_extractor>{match}</subtitle_extractor> are:\n" + str(cur_subtitle_triples) + '\n')
                except Exception as e:
                    print(f"Error processing subtitle extractor: {e}")
                    continue
        
        return ''.join(results)

    def remove_duplicate_sentences(self, text: str) -> str:
        
        if not text:
            return text
        
        sentences = re.split(r'[.。\n]+', text)
        
        seen = set()
        unique_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        return '. '.join(unique_sentences)



def process_dic(dic, args):
    try: 
        video_name = dic['video_path'].split('/')[-1][:-4]
        folder_path = f'{args.dataset_folder}/{args.dataset}/embeddings/{args.clip_duration}/large/'
        embedding_path = f'{folder_path}/{video_name}.pkl'
        if not os.path.exists(embedding_path):
            return None

        total_embeddings = pickle.load(open(embedding_path,'rb'))
        if len(total_embeddings) <= dic['duration'] // args.clip_duration - 3:
            return None
        clip_save_folder = f'{args.dataset_folder}/{args.dataset}/clips/{args.clip_duration}/{video_name}/'

        if os.path.exists(embedding_path) and os.path.exists(f'{folder_path}/{video_name}_clip_paths.pkl'):
            return dic
        else:
            print('no embedding')
            return None
    except Exception as e:
        print(e)
        return None

def check_valid_data(data_li, args):
    
    valid_data_li = []
    for dic in tqdm(data_li, desc='checking data'):
        dic = process_dic(dic, args)
        if dic:
            valid_data_li.append(dic)
    print(len(valid_data_li), len(data_li))
    return valid_data_li



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mlvu')
    parser.add_argument('--dataset_mode', type=str, default='')
    parser.add_argument('--dataset_folder', type=str, default='')
    parser.add_argument('--read_path', type=str)
    parser.add_argument('--clip_duration', type=int, default=10)
    parser.add_argument('--topk_per_query', type=int, default=1)
    parser.add_argument('--retriever_type', type=str, default='large')
    parser.add_argument('--thread_idx', type=int, default=0)
    parser.add_argument('--thread_num', type=int, default=1)
    parser.add_argument('--begin_sample_number', type=int, default=0)
    parser.add_argument('--end_sample_number', type=int, default=10000000000)
    parser.add_argument('--random_shuffle', action='store_true')
    parser.add_argument('--overwrite_output', type=int, default=0)
    parser.add_argument('--use_subtitle', type=int, default=0)
    parser.add_argument('--use_vllm', type=int, default=1)
    parser.add_argument('--clip_fps', type=float, default=2)
    parser.add_argument('--tasks', type=str, default='all')
    parser.add_argument('--max_workers', type=int, default=1) 
    parser.add_argument('--save_path', type=str, default='')
    args = parser.parse_args()

    
    vlm = os.getenv('API_MODEL_NAME_VLM').split('/')[-1]
    lrm = '_'.join(os.getenv('API_MODEL_NAME').split('/')[-2:])
    temp = '_'.join(os.getenv('API_MODEL_NAME_TEMPORAL_GROUNDING').split('/')[-2:])
    topk = os.getenv('TOPK')

    if args.max_workers!=1:
        print('Local server should set max_workers to 1!!!')
        args.max_workers=1

    if not args.save_path:
        args.save_path = f"./eval_result/{args.dataset}{args.dataset_mode}_LRM_{lrm}_TEMP_{temp}_VLM_{vlm}_TOPK_{topk}_CLIP_{args.clip_duration}.json.part{args.thread_idx}"
    else:
        args.save_path = args.save_path + f'.part{args.thread_idx}'
    os.makedirs('./eval_result', exist_ok=True)

    if args.dataset == 'LongVideoBench':
        args.use_subtitle = 1
    elif args.dataset in ['mlvu', 'lvbench']:
        args.use_subtitle = 0

    print(f"Save path: {args.save_path}")
    print(f"Max workers: {args.max_workers}")
    


    dataset_mode = args.dataset_mode

    read_path = f'{args.dataset_folder}/{args.dataset}/qa{dataset_mode}.json' 
    data_li = [dic for dic in json.load(open(read_path))]
    if args.tasks != 'all':
        data_li = [dic for dic in data_li if dic['task'] in args.tasks.split(',')]

    data_li = data_li[args.begin_sample_number:args.end_sample_number]
    data_li = check_valid_data(data_li, args)

    chunk_size = len(data_li) // args.thread_num + 1
    data_li = data_li[args.thread_idx * chunk_size:(args.thread_idx + 1) * chunk_size]

    print("Checking valid data...")
    print(f"Valid data count: {len(data_li)}")

    processed_results = []
    processed_questions = set()
    
    if os.path.exists(args.save_path) and not args.overwrite_output:
        try:
            base_path = args.save_path.split('.part')[0]+'.part*'
            all_files = glob.glob(base_path)
            for file in glob.glob(base_path+'*'):
                with open(file, 'r') as f:
                    cur_processed_results = json.load(f)['processed_results']
                    for dic in cur_processed_results:
                        if 'score' in dic and type(dic['score'])==bool and dic not in processed_results:
                            processed_results.append(dic)
            
            processed_questions = {
                dic['raw_data']['question'] + dic['raw_data']['video_path'] 
                for dic in processed_results
            }
            print(f"Loaded {len(processed_results)} previously processed samples")
        except Exception as e:
            print(f'Error loading existing results: {e}')
            processed_results = []
            processed_questions = set()
    
    unprocessed_data = []
    for dic in data_li:
        question_key = dic['question'] + dic['video_path']
        if question_key not in processed_questions:
            unprocessed_data.append(dic)
    
    print(f'Total samples: {len(data_li)}, Already processed: {len(processed_results)}, To process: {len(unprocessed_data)}')
    
    if not unprocessed_data:
        print("All samples already processed!")
        exit(0)

    if args.random_shuffle:
        random.shuffle(unprocessed_data)


    print(f"Starting multiprocessing with {args.max_workers} workers...")
    process_args = []
    for i, data_item in enumerate(unprocessed_data):
        process_args.append((args, data_item, i % args.max_workers))

    completed_count = 0
    total_count = len(unprocessed_data)
    
    manager = SingleSampleProcessor(args)
    correct_count = 0

    for dic in tqdm(unprocessed_data):
            # retry = 0
            # while retry < 3:
            # try:
                result = manager.process_single_sample(dic)
                if result:
                    processed_results.append(result)
                    completed_count += 1
                    if result.get('score', 0) > 0:
                        correct_count += 1

                print(f'Thread {args.thread_idx} Accuracy: {correct_count} / {completed_count}', correct_count / (completed_count + 1e-6) * 100)
                print(args.save_path)
                with open(args.save_path, 'w') as f:
                    json.dump({'experiences': [], 'processed_results': processed_results}, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                # break
                # except Exception as e:
                #     print(e)
        # if retry == 3:
        #     dic['pred_answer'] = 'A'
        #     result =  {
        #         'messages': [f'Error processing this sample: {e}'],
        #         'raw_data': dic,
        #         'score': dic['answer'] == dic['pred_answer']
        #     }
        #     processed_results.append()

    print(f"\nProcessing completed!")
    print(f"Total processed samples: {len(processed_results)}")
    
    if processed_results:
        correct_count = sum(1 for r in processed_results if r.get('score', 0) > 0)
        accuracy = correct_count / len(processed_results) * 100
        print(f"Final accuracy: {accuracy:.2f}% ({correct_count}/{len(processed_results)})")
        
        task_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
        for result in processed_results:
            if 'task' in result['raw_data']:
                task = result['raw_data']['task']
                if type(task) == list:
                    task = '\n'.join(task[0])
                task_stats[task]['total'] += 1
                if result.get('score', 0) > 0:
                    task_stats[task]['correct'] += 1
        
        if task_stats:
            print("\nTask-wise statistics:")
            for task, stats in task_stats.items():
                task_accuracy = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
                print(f"  {task}: {task_accuracy:.2f}% ({stats['correct']}/{stats['total']})")


