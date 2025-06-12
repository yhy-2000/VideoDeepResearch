import os
os.environ["VLLM_USE_MODELSCOPE"] = "false"   
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor 
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from video_utils  import _get_video_duration,_cut_video_clips,extract_subtitles,timestamp_to_clip_path,is_valid_video,is_valid_frame,extract_video_clip,parse_subtitle_time,clip_number_to_clip_path,image_paths_to_base64
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
from retriever import Retrieval_Manager
from prompt_seed15vl import *
from collections import defaultdict
import random
from qwen_vl_utils import process_vision_info
import time
from vllm import LLM, EngineArgs, SamplingParams
from openai import OpenAI
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
from moviepy.video.io.VideoFileClip import VideoFileClip
import pickle


MAX_DS_ROUND=20
os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.backends.cuda.matmul.allow_tf32 = True


class BatchGeniusManager:
    """批量推理管理器"""
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.ds_model_name = os.getenv('API_MODEL_NAME')
        self.ds_api_base = os.getenv('API_BASE_URL')
        self.ds_api_keys = [os.getenv('API_KEY')]

        self.vlm_model_name = os.getenv('API_MODEL_NAME_VLM')
        self.vlm_api_base = os.getenv('API_BASE_URL_VLM')
        self.vlm_api_key = os.getenv('API_KEY_VLM')


        self.processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct',use_fast=True)
        self.processor.tokenizer.padding_side = 'left'

        self.clip_save_folder = f'{self.args.dataset_folder}/{args.dataset}/clips/{args.clip_duration}/'
        self.retriever = Retrieval_Manager(args, clip_save_folder = self.clip_save_folder)
        
        gpu_id = args.thread_idx % 8 if len(os.getenv('CUDA_VISIBLE_DEVICES', '0').split(','))==8 else 0
        self.retriever.load_model_to_gpu(gpu_id)
        
        # 批量状态跟踪
        self.batch_states_text = {}
        self.processed_messages = []
        self.data_li = []

        self.current_batch = []
        self.current_sample_idx = 0
        self.batch_size = 100
        self.batch_size_vlm_vllm = 200 
        self.max_image_resolusion = 768
    

    def get_dic_subtitles(self, dic):
        video_id = dic['video_path'].split('/')[-1].split('.')[0]

        if os.path.exists(f'{self.args.benchmark_folder}/{self.args.dataset}/subtitles/{video_id}.srt'):
            subtitle_path = f'{self.args.benchmark_folder}/{self.args.dataset}/subtitles/{video_id}.srt'
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

                            subtitles += str(int(start_time)) + '-' + str(int(end_time)) +':'+ text + ' '
            dic['subtitles'] = subtitles
        elif os.path.exists(f'{self.args.dataset_folder}/{args.dataset}/subtitles/{video_id}.json'):
            subtitle_path = f'{self.args.dataset_folder}/{args.dataset}/subtitles/{video_id}.json'
            subtitles = json.load(open(subtitle_path))
            subtitles = [str(int(parse_subtitle_time(dic['start']))) + '-' +str(int(parse_subtitle_time(dic['end']))) +':'+ dic['line'] for dic in subtitles]
            dic[i]['subtitles'] = '\n'.join(subtitles)
        else:
            dic['subtitles'] = ''
        return dic

    def _init_batch_state(self, data: List[Dict], ans_li: List[Dict]):
        """初始化批量状态"""
        
        cur_question_li = [dic['question'] for dic in data]
        processed_question_li = []

        self.processed_messages = []
        for dic in ans_li:
            if dic['raw_data']['question'] in cur_question_li:
                self.processed_messages.append(dic)
                processed_question_li.append(dic['raw_data']['question'])
        
        self.data_li = [dic for dic in data if dic['question'] not in processed_question_li]
        
        print('@'*100)
        print('Current thread total samples:',len(data))
        print('processed samples:',len(self.processed_messages))
        print('left samples:',len(self.data_li))
        print('@'*100)

        batch_data = data[self.current_sample_idx:self.current_sample_idx + self.batch_size]
        self.current_sample_idx += len(batch_data)
        self.batch_states_text = {
            'raw_data': batch_data,
            'messages': [[] for _ in batch_data],
            'cur_turn': [0]*len(batch_data),
        }
        # 初始化消息提示
        for i, data in enumerate(batch_data):
            if self.args.use_subtitle:
                self.batch_states_text['raw_data'][i] = self.get_dic_subtitles(data)

            initial_prompt = self._build_initial_prompt(self.batch_states_text['raw_data'][i])
            self.batch_states_text['messages'][i].append({
                "role": "user",
                "content": [{"type": "text", "text": initial_prompt}]
            })
           
    def _build_initial_prompt(self, data: Dict) -> str:
        """构建初始提示模板"""
        # # 运动场景属于动作复杂型，候选clip需要更高fps避免丢失信息
        # if 'sport' in data['task'] or 'sport' in data["type"]:
        #     args.clip_duration = 5
        # else:
        #     args.clip_duration = 10

        if self.args.use_subtitle:
            if 'videomme' in self.args.dataset:
                base_prompt = initial_input_template_general_r1_subtitle_videomme.format(
                    question=data['question'] + "\n" + "\n".join(data['options']),
                    duration=data['duration'],
                    clip_duration=self.args.clip_duration,
                    subtitles = data['subtitles'],
                    MAX_DS_ROUND=MAX_DS_ROUND
                )
            else:
                base_prompt = initial_input_template_general_r1_subtitle_longvideobench.format(
                    question=data['question'] + "\n" + "\n".join(data['options']),
                    duration=data['duration'],
                    clip_duration=self.args.clip_duration,
                    MAX_DS_ROUND=MAX_DS_ROUND
                )

        else:
            if 'mlvu' in self.args.dataset:
                base_prompt = initial_input_template_general_r1_mlvu.format(
                    question=data['question'] + "\n" + "\n".join(data['options']),
                    duration=data['duration'],
                    clip_duration=self.args.clip_duration,
                    MAX_DS_ROUND=MAX_DS_ROUND
                )
            elif 'lvbench' in self.args.dataset:
                base_prompt = initial_input_template_general_r1_lvbench.format(
                    question=data['question'] + "\n" + "\n".join(data['options']),
                    duration=data['duration'],
                    clip_duration=self.args.clip_duration,
                    MAX_DS_ROUND=MAX_DS_ROUND
                )
            elif 'videomme' in  self.args.dataset:
                base_prompt = initial_input_template_general_r1_videomme.format(
                    question=data['question'] + "\n" + "\n".join(data['options']),
                    duration=data['duration'],
                    clip_duration=self.args.clip_duration,
                    MAX_DS_ROUND=MAX_DS_ROUND
                )
        return base_prompt


    def process(self, data: List[Dict], ans_li: List[Dict]):
        """处理批量数据主流程"""
        self._init_batch_state(data, ans_li)

        self.start_time = time.time()

        error = 0
        while len(self.processed_messages) + error < len(data):
            ok = 0
            retry = 0
            while not ok and retry<3:
                time2=time.time()
                print('batch_text2text........')
                ans_text = self.batch_text2text()
                time3=time.time()
                self._update_batch_context(ans_text)
                time4=time.time()
                print('tool outputs tackled, time:', int(time4-time3))
                ok=1
        
            error+=1
        

    def single_video2text(self, li):
        prompt, image_paths,timestamps = li
        vlm_api_base = self.vlm_api_base
        client = OpenAI(base_url=vlm_api_base, api_key=self.vlm_api_key)

        content = []
        for idx, image_path in enumerate(image_paths):
            base64_image = load_image(image_path)
            if base64_image==None:
                continue
            if timestamps is not None:
                # add timestamp for each frame
                content.append({
                    "type": "text",
                    "text": f'[{timestamps[idx]} second]'
                })
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail":"low"
                    },
                }
            )

        content.append(
            {
                "type": "text",
                "text": prompt,
        })
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        retry = 0
        while retry<5:
            try:
                ans=client.chat.completions.create(
                    model="doubao-1.5-vision-pro-250328",
                    messages=messages
                ).choices[0].message.content
                return ans
            except Exception as e:
                print(e, vlm_api_base)
                time.sleep(10)
                retry+=1



    def single_text2text(self, li):
        idx, message = li
        llm = OpenAI(base_url=self.ds_api_base, api_key=random.choice(self.ds_api_keys))
        retry = 0
        while retry<3:
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        lambda: llm.chat.completions.create(
                            model=self.ds_model_name,
                            messages=message,
                            # max_tokens=100000
                        )
                    )
                    completion = future.result(timeout=1800)  # 最多等待 300 秒
                    ans = completion.choices[0].message.content.strip('\n').strip()
                    return ans
            except:
                retry += 1
                time.sleep(10)


        self.batch_states_text['cur_turn'][idx]+=1


    def batch_text2text(self, thread_count=None):
        results = []
        with ThreadPoolExecutor(max_workers=min(self.batch_size,len(self.batch_states_text['messages']))) as executor:
            results = list(tqdm(executor.map(self.single_text2text, [(i,m) for i,m in enumerate(self.batch_states_text['messages'])]), total=len(self.batch_states_text['messages']),desc='calling deepseek'))

        return results


    def batch_video2text(self, query_list, image_list_list, timestamp_list_list, thread_count=None):
        print('calling seed1.5vl...')

        results = []
        with ThreadPoolExecutor(max_workers=min(self.batch_size_vlm_vllm,len(query_list))) as executor:
            results = list(tqdm(executor.map(self.single_video2text, [li for li in zip(query_list, image_list_list, timestamp_list_list)]), total=len(query_list),desc='calling deepseek'))

        return results

    def _update_batch_context(self, ans_text):
        torch.cuda.empty_cache()
        
        to_delete = {'messages': [], 'raw_data': [], 'cur_turn': []}
        batch_states=self.batch_states_text

        video_to_process, video_to_process_idx, total_valid_queries, total_valid_video_clips, total_valid_match_set, total_valid_timestamps = [], 0, [], [], [], []

        valid_flag_li = []
        for i, output_text in enumerate(ans_text):
            batch_states['messages'][i].append({'role':'assistant', 'content': output_text})
            valid_flag = 0

            video_path = batch_states['raw_data'][i]['video_path']
            duration = batch_states['raw_data'][i]['duration']

            if batch_states['cur_turn'][i] > MAX_DS_ROUND + 1:
                answer = self._extract_final_answer(output_text)
                dic = batch_states['raw_data'][i]
                score = answer[0] == dic['answer']
                del batch_states['messages'][i]['subtitles']
                self.processed_messages.append({'messages':batch_states['messages'][i], "raw_data":batch_states['raw_data'][i], "score":score})

                with open(self.args.save_path, "w") as fw:
                    json.dump(self.processed_messages, fw, indent=4)
                    fw.flush()

                # Mark for deletion instead of deleting immediately
                to_delete['messages'].append((batch_states, i))
                to_delete['raw_data'].append((batch_states, i))
                to_delete['cur_turn'].append((batch_states, i))
                valid_flag_li.append(0)
                continue

            tool_result = ''
            if "<video_reader_question>" in output_text:
                valid_flag = 1
                pattern = r"<video_reader>([^<]+)</video_reader>\s*<video_reader_question>([^<]+)</video_reader_question>"
                if '</thinking>' in output_text:
                    output_text = output_text.split('</thinking>')[1]
                matches = re.findall(pattern, output_text)

                # 提取question_matches 和 matches 的内容
                question_matches = [match[1] for match in matches]
                matches = [match[0] for match in matches]
                
                video_clip_folder = self.clip_save_folder + '/' +video_path.split('/')[-1].split('.')[0]
                max_clip_num = max([int(file.split('_')[1]) for file in os.listdir(video_clip_folder)])
                
                valid_queries, valid_video_clips, valid_timestamps, valid_match_set, vis = [],[],[],[],[]

                for query, match_set in zip(question_matches, matches):
                    if query+match_set not in vis:
                        if ':' in match_set:
                            begin_time_stamp, end_time_stamp = match_set.split(':')[0], match_set.split(':')[1]
                            begin_time_stamp, end_time_stamp = float(begin_time_stamp), float(end_time_stamp)
                            video_clip, timestamps = timestamp_to_clip_path(begin_time_stamp, end_time_stamp, video_path, fps=self.args.clip_fps)
                        else:
                            clip_numbers = sorted([int(m) for m in match_set.split(';') if m.isdigit()])
                            video_clip, timestamps = clip_number_to_clip_path(clip_numbers, video_path, clip_duration=self.args.clip_duration, fps = args.clip_fps)

                        query = (
                            "Please watch the given video and answer the following question: " + query +
                            "Output the detailed video description and the answer in this format: The description of the video is:YOUR_DESCRIPTION\nThe answer is:YOUR_ANSWER. If the question includes options, you may select one or multiple correct choices or none."
                        )
                        valid_queries.append(query)
                        valid_video_clips.append(video_clip)
                        valid_timestamps.append(timestamps)
                        valid_match_set.append(match_set)
                        vis.append(query+match_set)

                video_to_process.append([i, output_text, video_to_process_idx, video_to_process_idx + len(valid_queries)])
                total_valid_queries.extend(valid_queries)
                total_valid_video_clips.extend(valid_video_clips)
                total_valid_timestamps.extend(valid_timestamps)
                total_valid_match_set.extend(valid_match_set)
                video_to_process_idx = video_to_process_idx + len(valid_queries)
        
            

            if "<video_browser_question>" in output_text:
                valid_flag = 1
                pattern = r"<video_browser_question>([^<]+)</video_browser_question>"
                if '</thinking>' in output_text:
                    output_text = output_text.split('</thinking>')[1]
                query = re.findall(pattern, output_text)
                
                if len(query)==0:
                    continue
                else:
                    query=query[0]
                
                valid_queries, valid_video_clips, valid_timestamps, valid_match_set, vis = [],[],[],[],[]
                video_clip, timestamps = timestamp_to_clip_path(0, batch_states['raw_data'][i]['duration'], video_path, fps=self.args.clip_fps)

                ans = self.single_video2text([query, video_clip, timestamps])
                tool_result+=f"The tool results for <video_browser_question>{query}</video_browser_question> is:{ans}\n"  

                

            if '<video_segment_retriever_textual_query>' in output_text:
                valid_flag = 1
                pattern = r"<video_segment_retriever_textual_query>(.*?)</video_segment_retriever_textual_query>"
                if '</thinking>' in output_text:
                    output_text = output_text.split('</thinking>')[1]
                matches = re.findall(pattern, output_text, flags=re.DOTALL)

                topk_pattern = r"<topk>(.*?)</topk>"
                topk_matches = re.findall(topk_pattern, output_text, flags=re.DOTALL)
                video_paths = []
                for j,match_set in enumerate(matches):
                    for match in match_set.split(';'):
                        try:
                            topk = int(topk_matches[j]) 
                        except:
                            topk=5
                        video_clip_paths = self.retriever.get_informative_clips(match,video_path=video_path,top_k=topk,total_duration=batch_states['raw_data'][i]['duration'])
                        cur_video_paths = [int(video[0].split('/')[-1].split('_')[1]) for video in video_clip_paths]
                        tool_result+=f"The tool results for <video_segment_retriever_textual_query>{match}</video_segment_retriever_textual_query> are:\n"  + str(cur_video_paths) +'\n'


            if '<video_segment_retriever_image_query>' in output_text:
                valid_flag = 1
                pattern = r"<video_segment_retriever_image_query>(.*?)</video_segment_retriever_image_query>"
                if '</thinking>' in output_text:
                    output_text = output_text.split('</thinking>')[1]
                matches = re.findall(pattern, output_text, flags=re.DOTALL)

                topk_pattern = r"<topk>(.*?)</topk>"
                topk_matches = re.findall(topk_pattern, output_text, flags=re.DOTALL)
                video_paths = []
                for j,match_set in enumerate(matches):
                    for match in match_set.split(';'):
                        try:
                            topk = int(topk_matches[j])+1
                        except:
                            topk=10
                        begin,end = float(match)-1,float(match)+1
                        query_video_path = extract_video_clip(video_path,begin,end)
                        video_clip_paths = self.retriever.get_informative_clips_with_video_query(batch_states['raw_data'][i]["question_wo_referring_query"],query_video_path, video_path=video_path,top_k=topk,total_duration=batch_states['raw_data'][i]['duration'])
                        cur_video_paths = []
                        for video in video_clip_paths:
                            clip_number = int(video[0].split('/')[-1].split('_')[1])
                            if not clip_number*self.args.clip_duration <=float(match) <=clip_number*self.args.clip_duration + self.args.clip_duration:
                                cur_video_paths.append(clip_number)
                        tool_result+=f"The tool results for <video_segment_retriever_image_query>{match}</video_segment_retriever_image_query> are:\n"  + str(cur_video_paths) +'\n'

            if '<subtitle_retriever>' in output_text:
                valid_flag = 1
                pattern = r"<subtitle_retriever>(.*?)</subtitle_retriever>"
                
                if '</thinking>' in output_text:
                    output_text = output_text.split('</thinking>')[1]
                matches = re.findall(pattern, output_text, flags=re.DOTALL)

                # if len(matches)==0:
                #     batch_states['messages'][i].append({'role':'user', 'content': 'The output is invalid. You should strictly follow the provided xml format!!!'})
                #     continue

                topk_pattern = r"<topk>(.*?)</topk>"
                topk_matches = re.findall(topk_pattern, output_text, flags=re.DOTALL)

                for j,match_set in enumerate(matches):
                    subtitle_triples = []
                    vis = []
                    for match in match_set.split(';'):
                        topk = int(topk_matches[j]) if len(topk_matches)>j else 30
                        cur_subtitle_triples = self.retriever.get_informative_subtitles(match,video_path=video_path,top_k=topk,total_duration=batch_states['raw_data'][i]['duration'])
                        for x in cur_subtitle_triples:
                            if x[0] not in vis:
                                # 提取subtitle在抽取视频中的相对位置
                                if 'starting_timestamp_for_subtitles' in batch_states['raw_data'][i]:
                                    begin_timestamp = x[0]-batch_states['raw_data'][i]['starting_timestamp_for_subtitles']
                                    end_timestamp = x[1]-batch_states['raw_data'][i]['starting_timestamp_for_subtitles']
                                else:
                                    begin_timestamp, end_timestamp = x[0], x[1]
                                subtitle_triples.append({'begin_timestamp': begin_timestamp, 'end_timestamp': end_timestamp, 'text':x[2]})
                                vis.append(x[0])
                    subtitle_triples = sorted(subtitle_triples,key = lambda x:x['begin_timestamp'])
                    subtitles = '\n'.join([x['text'] for x in subtitle_triples])
                    tool_result+=f"The tool results for <subtitle_retriever>{match_set}</subtitle_retriever> are:\n"  + str(subtitle_triples) +'\n'
                    

            if '<subtitle_extractor>' in output_text:
                valid_flag = 1
                pattern = r"<subtitle_extractor>(.*?)</subtitle_extractor>"
                if '</thinking>' in output_text:
                    output_text = output_text.split('</thinking>')[1]
                matches = re.findall(pattern, output_text, flags=re.DOTALL)

                for match_set in matches:
                    for match in match_set.split(';'):
                        begin_timestamp, end_timestamp = float(match.split(':')[0]), float(match.split(':')[1])
                        if 'starting_timestamp_for_subtitles' in batch_states['raw_data'][i]:
                            begin_timestamp, end_timestamp = begin_timestamp + batch_states['raw_data'][i]['starting_timestamp_for_subtitles'], end_timestamp+ batch_states['raw_data'][i]['starting_timestamp_for_subtitles']

                        all_subtitle_triples = extract_subtitles(video_path)
                        cur_subtitle_triples = [{'begin_timestamp': int(x[0]), 'end_timestamp': int(x[1]), 'subtitle': x[2]} for x in all_subtitle_triples if begin_timestamp<=x[0]<=end_timestamp]
                        tool_result+=f"The tool results for <subtitle_extractor>{match}</subtitle_extractor> are:\n"  + str(cur_subtitle_triples) +'\n'

            if batch_states['cur_turn'][i] < MAX_DS_ROUND and valid_flag == 0 and '<answer>' not in output_text:
                print('#'*100)
                print('invalid output format!', output_text)
                print('#'*100)
                batch_states['messages'][i].append({'role':'user', 'content': 'The output is invalid. You should strictly follow the provided xml format!!!'})

            batch_states['cur_turn'][i] += 1
            if tool_result != '':
                batch_states['messages'][i].append({'role':'user', 'content': tool_result + f"You have now engaged in a total of {batch_states['cur_turn'][i]} rounds of conversation, with {10-batch_states['cur_turn'][i]} calls remaining. Please make the most of each opportunity until you obtain an accurate answer. Don't guess the answer!! Obtain an accurate answer with tools!!"})
            valid_flag_li.append(valid_flag)

        if len(total_valid_queries):
            total_ans_li = self.batch_video2text(total_valid_queries, total_valid_video_clips, total_valid_timestamps)

            for ii, output_text, begin_idx, end_idx in video_to_process:
                cur_ans_li = total_ans_li[begin_idx: end_idx]
                cur_valid_match_set = total_valid_match_set[begin_idx: end_idx]

                ans_str_li = [f'The tool result for <video_reader>{match}</video_reader> is {ans}' for ans, match in zip(cur_ans_li,cur_valid_match_set)]
                if batch_states['messages'][ii][-1]['role'] =='user':
                    batch_states['messages'][ii][-1]['content'] += '\n' + '\n'.join(ans_str_li)
                else:
                    batch_states['messages'][ii].append({'role':'user', 'content': '\n'.join(ans_str_li)})

        # 只有当没有工具调用时候的输出答案才算正确，不然都是蒙的
        for i, output_text in enumerate(ans_text):
            if '<answer>' in output_text and not valid_flag_li[i]:
                answer = self._extract_final_answer(output_text)
                dic = batch_states['raw_data'][i]
                score = answer[0] == dic['answer']
                print('score:', score)

                if 'subtitles' in batch_states['raw_data'][i]:
                    del batch_states['raw_data'][i]['subtitles']
                self.processed_messages.append({'messages':batch_states['messages'][i], "raw_data":batch_states['raw_data'][i], "score": score})

                self.current_time = time.time()
                print('current_sample_idx:',self.current_sample_idx, 'left time:', (self.current_time - self.start_time) / len(self.processed_messages)  * (len(self.data_li) - len(self.processed_messages)) / 3600, 'h')

                with open(self.args.save_path, "w") as fw:
                    json.dump(self.processed_messages, fw, indent=4)
                    fw.flush()
                print(self.args.save_path)

                if self.current_sample_idx < len(self.data_li):
                    dic = self.data_li[self.current_sample_idx]
                    self.current_sample_idx+=1

                    dic = self.get_dic_subtitles(dic)
                    initial_prompt = self._build_initial_prompt(dic)

                    batch_states['messages'][i]= [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": initial_prompt},
                            ]
                        }
                    ]
                    batch_states['cur_turn'][i] = 0
                    batch_states['raw_data'][i] = dic
                else:
                    to_delete['messages'].append((batch_states, i))
                    to_delete['raw_data'].append((batch_states, i))
                    to_delete['cur_turn'].append((batch_states, i))


        for i, output_text in enumerate(ans_text):
            role, user_response = batch_states['messages'][i][-1]['role'], batch_states['messages'][i][-1]['content']
            if role=='assistant' and '<answer>' in user_response:
                continue
            if role=='assistant' or user_response== '':
                print('idx:',i)
                print('ans_text:', ans_text[i])


        for i,output_text in enumerate(ans_text):
            if batch_states['cur_turn'][i] >= MAX_DS_ROUND and '<answer>' not in output_text:
                print('Maximum number of rounds reached!')
                if batch_states['messages'][i][-1]['role'] =='user':
                    batch_states['messages'][i][-1]['content'] += '\nMaximum number of rounds reached! Now you should output the final answer within <answer></answer>!!!'
                else:
                    batch_states['messages'][i].append({'role':'user', 'content': 'Maximum number of rounds reached! Now you should output the final answer within <answer></answer>!!!'})

        for key in to_delete:
            for batch_states, i in sorted(to_delete[key], key=lambda x: x[1], reverse=True):
                if i < len(batch_states[key]):
                    del batch_states[key][i]


    def _extract_final_answer(self, text: str) -> str:
        try:
            answer_content = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)[-1].strip()
            first_upper = re.search(r'[A-Z]', answer_content)
            return first_upper.group(0) if first_upper else '-'
        except:
            return '-'

def process_dic(dic, args):
    if not os.path.exists(dic['video_path']):
        return None
        
    video_name = dic['video_path'].split('/')[-1][:-4]
    folder_path = f'{args.dataset_folder}/{args.dataset}/embeddings/{args.clip_duration}/large/'
    embedding_path = f'{folder_path}/{video_name}.pkl'
    frame_folder = f'{args.dataset_folder}/{args.dataset}/dense_frames/{video_name}/'

    if not os.path.exists(frame_folder) or len(os.listdir(frame_folder))<dic['duration']*args.clip_fps-5:
        print('not enough frames', frame_folder)
        if os.path.exists(frame_folder):
            print(len(os.listdir(frame_folder)), dic['duration']*args.clip_fps-5)
        return None

    if os.path.exists(embedding_path) and os.path.exists(f'{folder_path}/{video_name}_clip_paths.pkl'):
        video_paths = pickle.load(open(f'{folder_path}/{video_name}_clip_paths.pkl', 'rb'))
        total_embeddings = pickle.load(open(embedding_path, 'rb'))
        return dic
    else:
        print('no embedding')
        return None



def check_valid_data(data_li):
    valid_data_li = []
    for dic in tqdm(data_li,desc = 'checking data'):
        dic = process_dic(dic,args)
        if dic:
            valid_data_li.append(dic)
    return valid_data_li


# 使用示例
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
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

    args = parser.parse_args()
    args.save_path = f'./eval_result/{args.dataset}{args.dataset_mode}_deepseek_clip_{args.clip_duration}.json.part{args.thread_idx}'
    os.makedirs('./eval_result/',exist_ok=True)

    # videomme可以自定义，其他写死
    if args.dataset == 'LongVideoBench':
        args.use_subtitle = 1
    elif args.dataset in ['mlvu', 'lvbench']:
        args.use_subtitle = 0


    print(args.save_path)
    # 初始化管理器
    all_thread_data=[]
    dataset_mode = args.dataset_mode

    read_path = f'{args.dataset_folder}/{args.dataset}/qa{dataset_mode}.json' 
    data_li = [dic for dic in json.load(open(read_path))]
    if args.tasks!='all':
        data_li = [dic for dic in data_li if dic['task'] in args.tasks.split(',')]

    data_li = data_li[args.begin_sample_number:args.end_sample_number]
    all_thread_data.extend(data_li)
    chunk_size = len(data_li) // args.thread_num +1
    data_li = data_li[args.thread_idx*chunk_size:(args.thread_idx+1)*chunk_size]

    data_li = check_valid_data(data_li)
    if os.path.exists(args.save_path) and not args.overwrite_output:
        try:
            question2dic = {dic['raw_data']['question']+dic['raw_data']['video_path']:dic for dic in json.load(open(args.save_path,'r'))}
            ans_li = [v for k,v in question2dic.items() if k in [dic['question']+dic['video_path'] for dic in data_li]]
        except Exception as e:
            print('cccccc',e)
            ans_li = []
    else:
        ans_li = []
    
    print('task number:', len(args.dataset_mode.split(',')), 'cur_thread_data_valid:',len(data_li))

    manager = BatchGeniusManager(args)
    manager.process(data_li, ans_li)
