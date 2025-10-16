import os
os.environ["VLLM_USE_MODELSCOPE"] = "false"   
import glob
import pickle
import threading
import fcntl
from vllm import LLM, EngineArgs, SamplingParams
import time
import re

file_lock = threading.Lock()


def safe_pickle_load(file_path):
    
    with file_lock:
        try:
            with open(file_path, 'rb') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading pickle file {file_path}: {e}")
            return None


def safe_pickle_dump(data, file_path):
    
    with file_lock:
        try:
            with open(file_path, 'wb') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"Error saving pickle file {file_path}: {e}")
            return False


def safe_file_remove(file_path):
    
    with file_lock:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")
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
                if os.path.isfile(file_path):
                    file_mtime = os.path.getmtime(file_path)
                    
                    if current_time - file_mtime > max_age_seconds:
                        safe_file_remove(file_path)
                        deleted_count += 1
                        print(f"Deleted old output file: {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
        if deleted_count > 0:
            print(f"Cleanup completed: deleted {deleted_count} old files from {output_directory}")
            
    except Exception as e:
        print(f"Error during cleanup: {e}")


def cleanup_worker(output_directory, interval_seconds=60):
    
    while True:
        try:
            cleanup_old_output_files(output_directory)
            time.sleep(interval_seconds)
        except Exception as e:
            print(f"Error in cleanup worker: {e}")
            time.sleep(interval_seconds)





def get_recently_modified_files(directory, num_files=30):
    all_files = glob.glob(os.path.join(directory, "**"), recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f) and 'lock' not in f and not os.path.exists(f'{f}.lock')]
    all_files.sort(key=os.path.getmtime, reverse=True)
    print('[PLAN] find files:', len(all_files))
    file_path_li, file_name_li, data_li = [],[],[]
    for file in all_files:
        try:
            dic = safe_pickle_load(file)
            if dic is None:
                continue
            if dic['model']!=MODEL_NAME:
                print(f"MODELNAME MISMATCH, {dic['model']}!={MODEL_NAME}")
                continue
            data_li.append(dic['input'])
            file_path_li.append(file)
            file_name_li.append(file.split('/')[-1])
            if len(data_li)>=num_files:
                break
        except Exception as e:
            print(e)

    return file_path_li,file_name_li, data_li

# MODEL_NAME= '/share/project/huaying/VideoDeepResearch/train/LLaMA-Factory-dpo/saves/qwen25-7b-dpo_1k/checkpoint-40/full_model'
MODEL_NAME= os.environ['API_MODEL_NAME']

class VLM_Listener:
    def __init__(self):
        
        self.output_directory = f'./vllm_io_files/vllm_output_planner'
        os.makedirs(self.output_directory, exist_ok=True)
        self.cleanup_thread = threading.Thread(
            target=cleanup_worker, 
            args=(self.output_directory, 60),  
            daemon=True
        )
        self.cleanup_thread.start()
        print("Started cleanup thread for output directory")

    
        try:
            print("Initializing VLLM server with conservative settings...")
            self.vlm_server = LLM(
                model = MODEL_NAME, 
                gpu_memory_utilization=0.85,  
                tensor_parallel_size=1,
                max_model_len=32768,  
                enable_chunked_prefill=True,
            )
            print("VLLM server initialized successfully")
        except Exception as e:
            print(f"Error initializing VLLM server: {e}")
            self.vlm_server = LLM(
                model = MODEL_NAME, 
                gpu_memory_utilization=0.9, 
                tensor_parallel_size=1,
                max_model_len=32768,  
                enable_chunked_prefill=True,
                enable_lora=True,
            )



    def run(self):

        
        while True:
            try:
                file_path_li,file_name_li, batch_inputs = get_recently_modified_files(f'./vllm_io_files/vllm_input_planner')
                if file_name_li==[]:
                    time.sleep(0.2)
                    continue

                for file in file_name_li:
                    input_dir = os.path.join('./vllm_io_files', 'vllm_input_planner')
                    safe_pickle_dump('lock', f"{input_dir}/{file}.lock")              

                print(f"[PLAN] Starting batch generation for {len(batch_inputs)} inputs...")

                batch_start_time = time.time()
                
                results_li = []
                try: 
                    sampling_params = SamplingParams(
                        temperature=0.0, 
                        max_tokens=8192,
                        skip_special_tokens=True,
                        stop_token_ids=None
                    )
                    outputs = self.vlm_server.chat(
                        batch_inputs,
                        sampling_params=sampling_params,
                        use_tqdm=False
                    )
                    
                    for i, (inputs, output) in enumerate(zip(batch_inputs,outputs)):
                        result_text = output.outputs[0].text
                        results_li.append(result_text)

                except Exception as e:
                    print(e)
                    results_li = ['']*len(batch_inputs)
                
                for i in range(len(results_li)):
                    safe_file_remove(file_path_li[i])
                    safe_file_remove(f"{file_path_li[i]}.lock")
                    safe_pickle_dump(results_li[i], f"{self.output_directory}/{file_name_li[i]}")
                batch_time = time.time() - batch_start_time
                print(f'Batch processing completed: {batch_time:.2f}s for {len(batch_inputs)} tasks')

            except Exception as e:
                print(e)
            
            


if __name__=='__main__':
    server = VLM_Listener()
    server.run()

