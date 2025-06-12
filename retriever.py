import os
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
from video_utils import *
import subprocess
import datetime
import multiprocessing

from FlagEmbedding import BGEM3FlagModel



class Retrieval_Manager():
    def __init__(self, args=None, batch_size=1, clip_save_folder=None, clip_duration=30):
        
        if args.retriever_type=='large':
            path = 'LanguageBind/LanguageBind_Video_FT'
        elif args.retriever_type=='huge':
            path = 'LanguageBind/LanguageBind_Video_Huge_V1.5_FT'
        else:
            raise KeyError

        clip_type = {
            'video':  path, # also LanguageBind_Video
            'image': 'LanguageBind/LanguageBind_Image'
        }

        self.model = LanguageBind(clip_type=clip_type, cache_dir='.')

        self.text_retriever = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation


        self.model.eval()

        self.tokenizer = LanguageBindVideoTokenizer.from_pretrained(path)
        self.modality_transform = {c: transform_dict[c](self.model.modality_config[c]) for c in clip_type.keys()}

        self.clip_embs_cache = {}
        self.frame_embs_cache = {}
        self.batch_size = 1
        self.clip_save_folder = clip_save_folder
        self.args=args


    def load_model_to_device(self, device):

        self.model.to(device)

        def recursive_to(module):
            for name, attr in module.__dict__.items():
                if isinstance(attr, torch.nn.Module):
                    attr.to(device)
                    recursive_to(attr)
                elif isinstance(attr, torch.Tensor):
                    setattr(module, name, attr.to(device))
                elif isinstance(attr, (list, tuple)):
                    new_attrs = []
                    for item in attr:
                        if isinstance(item, torch.nn.Module):
                            item.to(device)
                            recursive_to(item)
                        elif isinstance(item, torch.Tensor):
                            item = item.to(device)
                        new_attrs.append(item)
                    setattr(module, name, type(attr)(new_attrs))

        recursive_to(self.model)

    def load_model_to_cpu(self):
        self.device=torch.device('cpu')
        self.load_model_to_device(torch.device('cpu'))
    
    def load_model_to_gpu(self, gpu_id=0):
        self.device = torch.device(f'cuda:{gpu_id}')
        self.load_model_to_device(torch.device(f'cuda:{gpu_id}'))

    def cut_video(self, video_path, clip_save_folder=None, total_duration=-1):
        valid_clip_paths = set()
        time1 = time.time()
        os.makedirs(clip_save_folder, exist_ok=True)

        duration = VideoFileClip(video_path).duration
        chunk_number = math.ceil(duration/self.args.clip_duration)

        total_video_clip_paths = []
        for i in range(chunk_number):
            start_time = self.args.clip_duration * i
            end_time = start_time + self.args.clip_duration
            output_filename = f'clip_{i}_{self.format_time(start_time)}_to_{self.format_time(end_time)}.mp4'  
            total_video_clip_paths.append(clip_save_folder+'/'+output_filename)     

        if os.path.exists(clip_save_folder):
            retry = 0
            while retry < 2:
                valid_clip_num = 0
                for clip_name in os.listdir(clip_save_folder):
                    try:
                        VideoReader(clip_save_folder+'/'+clip_name, ctx=cpu(0), num_threads=1)
                        valid_clip_paths.add(clip_save_folder+'/'+clip_name)
                        valid_clip_num+=1
                        del total_video_clip_paths[total_video_clip_paths.index(clip_save_folder+'/'+clip_name)]
                    except Exception as e:
                        pass
                        
                if valid_clip_num >= (2*chunk_number//3): 
                    return [file for file in sorted(valid_clip_paths, key=lambda x: int(x.split('/')[-1].split('_')[1]))]
                else:
                    assert False,f'valid_clip_num:{valid_clip_num} < chunk_number-3: {chunk_number-3}, clip_save_folder:{clip_save_folder}'

            # 5次之后移除所有不合法的clip
            for path in total_video_clip_paths:
                try:
                    VideoReader(clip_save_folder+'/'+clip_name, ctx=cpu(0), num_threads=1)
                    valid_clip_num+=1
                except Exception as e:
                    os.system('rm -rf '+path)

        else:
            print(clip_save_folder,'no valid clips found, cutting video:', video_path)
        
        return sorted(list(valid_clip_paths), key=lambda x: int(x.split('/')[-1].split('_')[1]))

    def save_clip(self, clip, clip_save_folder, clip_index, start_time, end_time, fps):
        start_time_str = self.format_time(start_time)
        end_time_str = self.format_time(end_time)
        os.makedirs(clip_save_folder,exist_ok=True)
        clip_path = os.path.join(clip_save_folder, f"clip_{clip_index}_{start_time_str}_to_{end_time_str}.mp4")
        height, width, _ = clip[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

        for frame in clip:
            out.write(frame)

        out.release()
        return clip_path

    def format_time(self, seconds):
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        return f"{int(hours):02d}-{int(mins):02d}-{int(secs):02d}"

    def parse_time(self, time_str):
        hours, mins, secs = map(int, time_str.split('-'))
        total_seconds = hours * 3600 + mins * 60 + secs
        return total_seconds


    @ torch.no_grad()
    def calculate_video_clip_embedding(self, video_path, folder_path, total_duration=None):
        total_embeddings = []
        video_name = video_path.split('/')[-1].split('.')[0]

        folder_path = f'{self.args.dataset_folder}/embeddings/{self.args.clip_duration}/{self.args.retriever_type}/'
        os.makedirs(folder_path,exist_ok=True)

        embedding_path = os.path.join(folder_path,video_name+'.pkl')
        clip_path = os.path.join(folder_path,video_name+'_clip_paths.pkl')

        if os.path.exists(embedding_path) and os.path.exists(clip_path):
            video_paths = pickle.load(open(clip_path,'rb'))
            total_embeddings = pickle.load(open(embedding_path,'rb'))
        
            invalid_num, invalid_videos=0,[]
            for v in video_paths:
                if not is_valid_video(v):
                    invalid_num+=1
                    invalid_videos.append(v)

            if invalid_num<3:
                return video_paths, total_embeddings
            else:
                print(embedding_path,'exist but have not enough valid video number!!',invalid_videos[0])
        # print('calculating video embeddings')
        video_paths = self.cut_video(video_path, os.path.join(self.clip_save_folder,video_path.split('/')[-1].split('.')[0]),total_duration)

        p = os.path.join(self.clip_save_folder,video_path.split('/')[-1].split('.')[0])
        assert len(video_paths) != 0, f'folder {p} have no valid clips'

        total_embeddings = []
        valid_video_paths = []
        for i in range(len(video_paths)):
            try:
                inputs = {'video': to_device(self.modality_transform['video'](video_paths[i]), self.device)}
                with torch.no_grad():
                    embeddings = self.model(inputs)
                    valid_video_paths.append(video_paths[i])
                    total_embeddings.append(embeddings['video'])
            except Exception as e:
                print(e)
            torch.cuda.empty_cache()
        total_embeddings = torch.cat(total_embeddings,dim=0)
        os.makedirs(folder_path,exist_ok=True)
        pickle.dump(total_embeddings,open(f'{folder_path}/{video_name}.pkl','wb'))
        pickle.dump(valid_video_paths,open(f'{folder_path}/{video_name}_clip_paths.pkl','wb'))
        return video_paths,total_embeddings




    def extract_frames(self, video_path, output_dir, fps=1):
        os.makedirs(output_dir, exist_ok=True)
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            print(f"Failed to open {video_path}")
            return
        
        frame_rate = vid.get(cv2.CAP_PROP_FPS)
        if frame_rate == 0:
            print(f"Failed to get FPS for {video_path}")
            return
        
        frame_interval = math.floor(frame_rate / fps)
        frame_idx = 0
        second = 0
        
        with tqdm(total=int(vid.get(cv2.CAP_PROP_FRAME_COUNT)), desc=os.path.basename(video_path)) as pbar:
            while True:
                ret, frame = vid.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    frame_filename = os.path.join(output_dir, f"frame_{second}.png")
                    cv2.imwrite(frame_filename, frame)
                    second += 1
                
                frame_idx += 1
                pbar.update(1)
        
        vid.release()

    @ torch.no_grad()
    def calculate_frame_embedding(self, video_path, folder_path, total_duration):
        total_embeddings = []
        video_name = video_path.split('/')[-1].split('.')[0]
        embedding_path = f'{folder_path}/{video_name}.pkl'
        
        if os.path.exists(embedding_path):
            os.makedirs(f'{args.dataset_folder}/embeddings/frame/{self.args.retriever_type}/',exist_ok=True)
            frame_paths = pickle.load(open(f'{folder_path}/{video_name}_frame_paths.pkl','rb'))
            total_embeddings = pickle.load(open(embedding_path,'rb'))
            invalid_num=0
            for v in frame_paths:
                if not is_valid_video(v):
                    invalid_num+=1

            if invalid_num<5:
                return frame_paths,total_embeddings
            
        frame_folder = '/'.join(video_path.split('/')[:-2]) + '/dense_frames/' + video_path.split('/')[-1].split('.')[0] + '/'
        if not os.path.exists(frame_folder) or os.listdir(frame_folder)==[]:
            self.extract_frames(video_path, frame_folder, fps=1)
        frame_paths = [frame_folder + file for file in sorted(os.listdir(frame_folder),key = lambda x:float(x.split('/')[-1].split('_')[1].split('.')[0]))]

        p = os.path.join(self.clip_save_folder,video_path.split('/')[-1].split('.')[0])
        assert len(frame_paths) != 0, f'folder {p} have no valid clips'

        total_embeddings = []
        valid_frame_paths = []
        for i in range(len(frame_paths)):
            try:
                inputs = {'image': to_device(self.modality_transform['image'](frame_paths[i]), self.device)}
                with torch.no_grad():
                    embeddings = self.model(inputs)
                    valid_frame_paths.append(frame_paths[i])
                    total_embeddings.append(embeddings['image'])
            except:
                pass
            torch.cuda.empty_cache()
        total_embeddings = torch.cat(total_embeddings,dim=0)
        os.makedirs(folder_path,exist_ok=True)
        pickle.dump(total_embeddings,open(f'{folder_path}/{video_name}.pkl','wb'))
        pickle.dump(valid_frame_paths,open(f'{folder_path}/{video_name}_frame_paths.pkl','wb'))
        return frame_paths,total_embeddings



    @ torch.no_grad()
    def calculate_video_embedding(self, video_path, folder_path):
        video_name = video_path.split('/')[-1].split('.')[0]
        os.makedirs(folder_path,exist_ok=True)
        embedding_path = f'{folder_path}/{video_name}.pkl'
        
        if os.path.exists(embedding_path):
            try:
                embedding = pickle.load(open(embedding_path,'rb'))
                return embedding
            except:
                pass

        try: 
            inputs = {'video': to_device(self.modality_transform['video'](video_path), self.device)}
            with torch.no_grad():
                embedding = self.model(inputs)
            pickle.dump(embedding,open(f'{folder_path}/{video_name}.pkl','wb'))
            return embedding
        except Exception as e:
            print(e)
            torch.cuda.empty_cache()



    @ torch.no_grad()
    def calculate_text_embedding(self,text,video_path=None,flag_save_embedding=True):
        if flag_save_embedding:
            video_name = video_path.split('/')[-1].split('.')[0]
            os.makedirs(f'{self.args.dataset_folder}/embeddings/subtitle/{self.args.retriever_type}',exist_ok=True)
            embedding_path = f'{self.args.dataset_folder}/embeddings/subtitle/{self.args.retriever_type}/{video_name}_subtitle.pkl'
            try:
                embeddings = pickle.load(open(embedding_path,'rb'))
                # print('use precalculated subtitle embeddings')
                return embeddings
            except:
                pass

        # print('calculating subtitle embeddings')
        inputs = {'language':to_device(self.tokenizer(text, max_length=77, padding='max_length',truncation=True, return_tensors='pt'), self.device)}

        with torch.no_grad():
            embeddings = self.model(inputs)
        if flag_save_embedding:
            pickle.dump(embeddings['language'],open(embedding_path,'wb'))
        torch.cuda.empty_cache()
        return embeddings['language']


    @ torch.no_grad()
    def calculate_subtitle_embedding(self,video_path,flag_save_embedding=False,merge_sentence=False):
        subtitles_with_time = extract_subtitles(video_path)
        subtitles = [x[2] for x in subtitles_with_time]
        subtitle_embs = self.calculate_text_embedding(subtitles,video_path,flag_save_embedding=True)
        subtitle_embs = subtitle_embs.cpu()
        return subtitles_with_time,subtitle_embs


    @ torch.no_grad()
    def get_informative_subtitles(self, query, video_path, top_k=1, total_duration=-1, return_embeddings=False,merge_sentence=False,flag_save_embedding=1):
        if not os.path.exists(video_path.replace('videos','subtitles').replace('.mp4','.srt')) and not os.path.exists(video_path.replace('videos','subtitles').replace('.mp4','_en.json')):
            return ''

        q_emb = self.text_retriever.encode(query, batch_size=12, max_length=256)['dense_vecs']
        subtitles_with_time = extract_subtitles(video_path)
        subtitles = [x[2] for x in subtitles_with_time]

        if flag_save_embedding:
            video_name = video_path.split('/')[-1].split('.')[0]
            os.makedirs(f'{self.args.dataset_folder}/embeddings/subtitle/{self.args.retriever_type}',exist_ok=True)
            embedding_path = f'{self.args.dataset_folder}/embeddings/subtitle/{self.args.retriever_type}/{video_name}_subtitle.pkl'
            try:
                subtitle_embeddings = pickle.load(open(embedding_path,'rb'))
            except Exception as e:
                print(e)
                subtitle_embeddings = self.text_retriever.encode(subtitles, batch_size=12, max_length=256)['dense_vecs']
                if flag_save_embedding:
                    pickle.dump(subtitle_embeddings,open(embedding_path,'wb'))

        similarities = np.dot(q_emb, subtitle_embeddings.T).flatten()  # shape: (832,)
        top_k_indices = np.argsort(similarities)[-top_k:][::-1].tolist()
        return [subtitles_with_time[i] for i in top_k_indices]



    def subtitle2clips(self, subtitle_triple, video_path):
        def is_overlap(begin1, end1, begin2, end2):
            return begin1 <= end2 and begin2 <= end1

        subtitle_begin_time, subtitle_end_time = subtitle_triple[0], subtitle_triple[1]
        ans = []
        for clip in os.listdir(self.clip_save_folder + video_path.split('/')[-1][:-4]):
            clip_begin_time, clip_end_time = self.parse_time(clip.split('.')[0].split('_')[2]),self.parse_time(clip.split('.')[0].split('_')[4])
            if is_overlap(subtitle_begin_time, subtitle_end_time, clip_begin_time, clip_end_time):
                video_clip_path = self.clip_save_folder + video_path.split('/')[-1][:-4] +f'/{clip}'
                ans.append(video_clip_path)
        return ans

    @ torch.no_grad()
    def get_informative_clips_with_video_query(self,query, query_video_path,video_path,top_k=0,similarity_threshold=-100,topk_similarity=0,total_duration=-1,return_score=False):
        torch.cuda.empty_cache()
        assert top_k!=0 and similarity_threshold==-100 and topk_similarity==0 or top_k==0 and similarity_threshold!=-100 and topk_similarity==0 or top_k==0 and similarity_threshold==-100 and topk_similarity!=0,f'only one of top_k and simlarity_threshold should be assigned!'

        if similarity_threshold!=-100 or topk_similarity!=0:
            top_k=100

        # Calculate and normalize the query embedding
        text_emb = self.calculate_text_embedding(query,flag_save_embedding=False).cpu()
        text_emb = text_emb / text_emb.norm(p=2, dim=1, keepdim=True)

        inputs = {'video': to_device(self.modality_transform['video'](query_video_path), self.device)}
        with torch.no_grad():
            q_emb = self.model(inputs)['video'].cpu()
        q_emb = q_emb / q_emb.norm(p=2, dim=1, keepdim=True)

        q_emb = q_emb + text_emb

        if video_path not in self.clip_embs_cache:
            if len(self.clip_embs_cache) > 1:  # Only keep cache for one video
                self.clip_embs_cache = {}
            video_name = video_path.split('/')[-1].split('.')[0]
            folder_path = f'{args.dataset_folder}/embeddings/{self.args.clip_duration}/{self.args.retriever_type}'
            video_clip_paths, clip_embs = self.calculate_video_clip_embedding(video_path, folder_path, total_duration)
            if type(clip_embs)==dict:
                clip_embs = clip_embs['video']

            clip_embs = clip_embs.cpu()
            self.clip_embs_cache[video_path] = video_clip_paths, clip_embs
        else:
            video_clip_paths, clip_embs = self.clip_embs_cache[video_path]

        # Normalize the clip embeddings
        clip_embs = clip_embs / clip_embs.norm(p=2, dim=1, keepdim=True)

        # Compute similarities
        similarities = torch.matmul(q_emb, clip_embs.T)

        # Get the indices of the top_k clips
        top_k_indices = similarities[0].argsort(descending=True)[:top_k].tolist()

        # Return list of tuples (path, similarity score) with similarity above threshold
        result = []
        
        for i in top_k_indices:
            sim_score = similarities[0][i].item()
            # print(sim_score)
            if sim_score > similarity_threshold:
                result.append((video_clip_paths[i], sim_score))
        
        torch.cuda.empty_cache()
        if top_k==0:
            result = result[:10] # 最多10个clip
        return result



    @ torch.no_grad()
    def get_informative_clips(self,query,video_path,top_k=0,similarity_threshold=-100,topk_similarity=0,total_duration=-1,return_score=False):
        torch.cuda.empty_cache()
        assert top_k!=0 and similarity_threshold==-100 and topk_similarity==0 or top_k==0 and similarity_threshold!=-100 and topk_similarity==0 or top_k==0 and similarity_threshold==-100 and topk_similarity!=0,f'only one of top_k and simlarity_threshold should be assigned!'

        if similarity_threshold!=-100 or topk_similarity!=0:
            top_k=100

        # Calculate and normalize the query embedding
        q_emb = self.calculate_text_embedding(query,flag_save_embedding=False).cpu()
        q_emb = q_emb / q_emb.norm(p=2, dim=1, keepdim=True)

        if video_path not in self.clip_embs_cache:
            if len(self.clip_embs_cache) > 1:  # Only keep cache for one video
                self.clip_embs_cache = {}
            video_name = video_path.split('/')[-1].split('.')[0]
            folder_path = f'{self.args.dataset_folder}/embeddings/{self.args.clip_duration}/{self.args.retriever_type}'
            video_clip_paths, clip_embs = self.calculate_video_clip_embedding(video_path, folder_path, total_duration)
            if type(clip_embs)==dict:
                clip_embs = clip_embs['video']

            clip_embs = clip_embs.cpu()
            self.clip_embs_cache[video_path] = video_clip_paths, clip_embs
        else:
            video_clip_paths, clip_embs = self.clip_embs_cache[video_path]

        # Normalize the clip embeddings
        clip_embs = clip_embs / clip_embs.norm(p=2, dim=1, keepdim=True)

        # Compute similarities
        similarities = torch.matmul(q_emb, clip_embs.T)

        # Get the indices of the top_k clips
        top_k_indices = similarities[0].argsort(descending=True)[:top_k].tolist()

        # Return list of tuples (path, similarity score) with similarity above threshold
        result = []
        
        for i in top_k_indices:
            sim_score = similarities[0][i].item()
            # print(sim_score)
            if sim_score > similarity_threshold:
                result.append((video_clip_paths[i], sim_score))
        
        torch.cuda.empty_cache()
        if top_k==0:
            result = result[:10] # 最多10个clip
        return result
