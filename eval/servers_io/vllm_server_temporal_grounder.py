import os
os.environ["VLLM_USE_MODELSCOPE"] = "false"   

import glob
import pickle
import threading
import fcntl
import uuid
import time
import re
import logging
from vllm import LLM, EngineArgs, SamplingParams

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

file_lock = threading.Lock()


file_lock = threading.Lock()
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


def safe_file_remove(file_path):
    
    with file_lock:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except Exception as e:
            logger.error(f"Error removing file {file_path}: {e}")
            return False

def remove_duplicate_sentences(text: str) -> str:
    
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

def cleanup_old_output_files(output_directory, max_age_minutes=5):
    
    try:
        current_time = time.time()
        max_age_seconds = max_age_minutes * 60
        
        pattern = os.path.join(output_directory, "*")
        files = glob.glob(pattern)
        
        deleted_count = 0
        for file_path in files:
            try:
                if os.path.isfile(file_path) and not file_path.endswith('.tmp'):
                    file_mtime = os.path.getmtime(file_path)
                    if current_time - file_mtime > max_age_seconds:
                        safe_file_remove(file_path)
                        deleted_count += 1
                        logger.info(f"Deleted old output file: {file_path}")
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleanup completed: deleted {deleted_count} old files from {output_directory}")
            
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def cleanup_old_lock_files(input_directory, max_age_minutes=2):
    
    try:
        current_time = time.time()
        max_age_seconds = max_age_minutes * 60
        
        pattern = os.path.join(input_directory, "*.lock")
        lock_files = glob.glob(pattern)
        
        deleted_count = 0
        for lock_file in lock_files:
            try:
                if os.path.isfile(lock_file):
                    file_mtime = os.path.getmtime(lock_file)
                    if current_time - file_mtime > max_age_seconds:
                        safe_file_remove(lock_file)
                        deleted_count += 1
                        logger.info(f"Deleted stale lock file: {lock_file}")
            except Exception as e:
                logger.error(f"Error processing lock file {lock_file}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Lock cleanup completed: deleted {deleted_count} stale lock files")
            
    except Exception as e:
        logger.error(f"Error during lock cleanup: {e}")

def cleanup_worker(output_directory, input_directory, interval_seconds=120):
    
    while True:
        try:
            cleanup_old_output_files(output_directory)
            cleanup_old_lock_files(input_directory)
            time.sleep(interval_seconds)
        except Exception as e:
            logger.error(f"Error in cleanup worker: {e}")
            time.sleep(interval_seconds)

def get_recently_modified_files(directory, num_files=30):
    
    all_files = glob.glob(os.path.join(directory, "**"), recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f) and 'lock' not in f and not os.path.exists(f'{f}.lock')]
    all_files.sort(key=os.path.getmtime, reverse=True)
    print('[TEMP] find files:', len(all_files))

    file_path_li, file_name_li, data_li = [], [], []
    process_id = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    
    for file in all_files:
        if len(data_li) >= num_files:
            break
            
        lock_file_path = f"{file}.lock"
        
        try:
            lock_fd = os.open(lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(lock_fd, f"{process_id}\n{time.time()}".encode())
                os.close(lock_fd)
                
                if not os.path.exists(file):
                    safe_file_remove(lock_file_path)
                    continue
                
                dic = safe_read_with_lock(file)
                if dic is None:
                    safe_file_remove(lock_file_path)
                    continue
                    
                if dic['model'] != MODEL_NAME:
                    logger.warning(f"MODELNAME MISMATCH, {dic['model']}!={MODEL_NAME}")
                    safe_file_remove(lock_file_path)
                    continue
                    
                data_li.append(dic['input'])
                file_path_li.append(file)
                file_name_li.append(file.split('/')[-1])
                
            except Exception as e:
                try:
                    os.close(lock_fd)
                except:
                    pass
                safe_file_remove(lock_file_path)
                logger.error(f"Error processing file {file}: {e}")
                
        except FileExistsError:
            continue
        except Exception as e:
            logger.error(f"Error creating lock for {file}: {e}")
            continue

    return file_path_li, file_name_li, data_li

MODEL_NAME='/share/project/huaying/VideoDeepResearch/train/LLaMA-Factory-main/saves/qwen25-7b-sft-temporal_grounding_real_6k_base'

class VLM_Listener:
    def __init__(self):
        self.process_id = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        logger.info(f"Initializing Temporal VLM_Listener with process ID: {self.process_id}")
        
        self.input_directory = f'./vllm_io_files/vllm_input_temporal'
        self.output_directory = f'./vllm_io_files/vllm_output_temporal'
        os.makedirs(self.output_directory, exist_ok=True)
        self.cleanup_thread = threading.Thread(
            target=cleanup_worker, 
            args=(self.output_directory, self.input_directory, 120),
            daemon=True
        )
        self.cleanup_thread.start()
        logger.info("Started cleanup thread for output directory")

        try:
            logger.info("Initializing VLLM server with conservative settings...")
            self.vlm_server = LLM(
                model = MODEL_NAME, 
                gpu_memory_utilization=0.85,
                tensor_parallel_size=1,
                max_model_len=32768,  
                enable_chunked_prefill=True,
            )
            logger.info("VLLM server initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing VLLM server: {e}")
            raise

    def run(self):
        consecutive_empty_counts = 0
        max_empty_before_sleep = 10
        
        while True:
            try:
                file_path_li, file_name_li, batch_inputs = get_recently_modified_files(self.input_directory)
                
                if file_name_li == []:
                    consecutive_empty_counts += 1
                    if consecutive_empty_counts >= max_empty_before_sleep:
                        logger.debug(f'[{self.process_id}] No tasks found, sleeping...')
                        time.sleep(2)
                        consecutive_empty_counts = 0
                    else:
                        time.sleep(1)
                    continue

                consecutive_empty_counts = 0
                logger.info(f"[{self.process_id}] Starting batch generation for {len(batch_inputs)} inputs...")

                batch_start_time = time.time()
                results_li = []
                
                try: 
                    sampling_params = SamplingParams(
                        temperature=0.0, 
                        max_tokens=32768,
                        skip_special_tokens=True,
                        stop_token_ids=None
                    )
                    
                    outputs = self.vlm_server.chat(
                        batch_inputs,
                        sampling_params=sampling_params,
                        use_tqdm=False
                    )
                    
                    for i, (inputs, output) in enumerate(zip(batch_inputs, outputs)):
                        result_text = output.outputs[0].text
                        result_text = remove_duplicate_sentences(result_text)
                        results_li.append(result_text)

                except Exception as e:
                    logger.error(f"[{self.process_id}] Error during batch generation: {e}")
                    results_li = [''] * len(batch_inputs)
                
                for i in range(len(results_li)):
                    try:
                        safe_file_remove(file_path_li[i])
                        safe_file_remove(f"{file_path_li[i]}.lock")
                        
                        output_path = f"{self.output_directory}/{file_name_li[i]}"
                        safe_write_with_lock(results_li[i], output_path)
                        
                    except Exception as e:
                        logger.error(f"Error saving result for {file_name_li[i]}: {e}")

                batch_time = time.time() - batch_start_time
                logger.info(f'[{self.process_id}] Batch processing completed: {batch_time:.2f}s for {len(batch_inputs)} tasks')
                
            except KeyboardInterrupt:
                logger.info(f"[{self.process_id}] Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"[{self.process_id}] Error in main loop: {e}")
                try:
                    if 'file_path_li' in locals():
                        for file_path in file_path_li:
                            safe_file_remove(f"{file_path}.lock")
                except:
                    pass
                time.sleep(5)

if __name__=='__main__':
    try:
        server = VLM_Listener()
        server.run()
    except Exception as e:
        logger.error(f"Failed to start VLM_Listener: {e}")
        raise