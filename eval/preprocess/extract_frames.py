import os
import cv2
import multiprocessing as mp
from multiprocessing import Pool
import glob
from pathlib import Path
import time
from tqdm import tqdm

def extract_frames_from_video(video_path):
    
    output_base_dir = f"./benchmark/{args.dataset}/dense_frames"
    
    video_name = Path(video_path).stem
    video_output_dir = os.path.join(output_base_dir, video_name)
    
    os.makedirs(video_output_dir, exist_ok=True)
    
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在 {video_path}")
        return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频 {video_path}")
        return False
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps if original_fps > 0 else 0
    
    
    target_fps = 2
    frame_interval = int(original_fps / target_fps) if original_fps > 0 else 1
    if frame_interval == 0:
        frame_interval = 1
    
    frame_count = 0
    saved_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                timestamp = frame_count / original_fps if original_fps > 0 else frame_count * 0.2
                
                frame_filename = f"frame_{timestamp:.2f}.png"
                frame_path = os.path.join(video_output_dir, frame_filename)
                
                success = cv2.imwrite(frame_path, frame)
                if success:
                    saved_count += 1
                else:
                    print(f"  - 保存失败: {frame_path}")
            
            frame_count += 1
    
    except Exception as e:
        print(f"处理视频时出错 {video_path}: {str(e)}")
        return False
    
    finally:
        cap.release()
    
    print(f"完成处理: {video_name}, 共保存 {saved_count} 帧")
    return True


def main():
    input_dir = f"./benchmark/{args.dataset}/videos"
    output_dir = f"./benchmark/{args.dataset}/dense_frames"
    
    print(f"=== 视频帧提取工具 ===")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"提取帧率: 5 fps")
    print(f"帧命名格式: frame_x.xx.png")
    
    os.makedirs(output_dir, exist_ok=True)
    
    video_pattern = os.path.join(input_dir, "*.mp4")
    video_files = glob.glob(video_pattern)
    
    if not video_files:
        print(f"在 {input_dir} 中没有找到 *.mp4 文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    processes = 96
    print(f"使用 {processes} 个进程并行处理")
    
    start_time = time.time()
    
    print(f"\n开始处理...")
    
    with Pool(processes=processes) as pool:
        results = tqdm(list(pool.map(extract_frames_from_video, video_files)))
    
    successful = sum(results)
    failed = len(results) - successful
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n=== 处理完成 ===")
    print(f"总视频数: {len(video_files)}")
    print(f"成功处理: {successful}")
    print(f"处理失败: {failed}")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"平均每个视频: {elapsed_time/len(video_files):.2f} 秒")
    
    if failed > 0:
        print(f"\n警告: 有 {failed} 个视频处理失败，请检查日志")
    
    print(f"\n输出目录结构:")
    if os.path.exists(output_dir):
        subdirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        print(f"创建了 {len(subdirs)} 个子目录")
        if len(subdirs) > 0:
            for i, subdir in enumerate(subdirs[:5]):
                subdir_path = os.path.join(output_dir, subdir)
                frame_count = len([f for f in os.listdir(subdir_path) if f.endswith('.png')])
                print(f"  {subdir}: {frame_count} 帧")
            if len(subdirs) > 5:
                print(f"  ... 还有 {len(subdirs) - 5} 个目录")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MLVU视频帧提取工具")
    parser.add_argument('--dataset', type=str, default='mlvu', help='数据集名称，默认为mlvu')
    args = parser.parse_args()
    
    main()
