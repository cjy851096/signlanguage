import cv2
import os
import numpy as np
import json
from ultralytics import YOLO
from tqdm import tqdm
import torch

# 1. 环境初始化
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolo11n-pose.pt').to(device)

def process_custom_json_to_landmarks(video_dir, output_dir, json_path):
    # 获取磁盘文件集合
    all_files = os.listdir(video_dir)
    file_set = {f: f for f in all_files if f.endswith(('.mp4', '.avi'))}
    print(f"磁盘视频总数: {len(file_set)}")

    # 加载 JSON
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    video_ids = list(data.keys())
    print(f"JSON 中记录的视频数: {len(video_ids)}")

    if not os.path.exists(output_dir): 
        os.makedirs(output_dir, exist_ok=True)

    success_count = 0

    # 遍历 JSON 中的每一个 video_id
    for v_id in tqdm(video_ids, desc="处理进度"):
        # 1. 匹配磁盘文件 (处理 05237 和 5237 这种前导零的可能性)
        # 尝试原名, 补零到5位, 以及去零后的名字
        possible_keys = [v_id, v_id.zfill(5), v_id.lstrip('0')]
        target_file = None
        
        for pk in possible_keys:
            fname = f"{pk}.mp4"
            if fname in file_set:
                target_file = fname
                break
        
        if not target_file:
            continue # 没找到文件则跳过

        # 获取类别标签 (使用 action 中的第一个元素作为类别 ID)
        action_info = data[v_id].get('action', [])
        label = str(action_info[0]) if action_info else "unknown"

        video_path = os.path.join(video_dir, target_file)
        cap = cv2.VideoCapture(video_path)
        
        sequence_data = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # 推理
            results = model(frame, verbose=False, conf=0.25, imgsz=320)
            
            if len(results) > 0 and results[0].keypoints is not None:
                # 获取关键点并转为 numpy
                kpts = results[0].keypoints.xyn[0].cpu().numpy()
                sequence_data.append(kpts)
            else:
                # 如果没检测到人，填充全 0 占位
                sequence_data.append(np.zeros((17, 2)))
        
        cap.release()

        # 保存结果
        if len(sequence_data) > 0:
            # 保存格式: 视频ID_类别ID.npy
            save_path = os.path.join(output_dir, f"{v_id}_{label}.npy")
            np.save(save_path, np.array(sequence_data))
            success_count += 1

    print(f"\n--- 处理完成！---")
    print(f"总计匹配并保存: {success_count} 个特征文件")

# 配置路径
VIDEO_DIR = r'E:\signlanguage\archive\videos' 
JSON_PATH = r'E:\signlanguage\archive\nslt_100.json'
OUTPUT_DIR = r'E:\signlanguage\output'

if __name__ == "__main__":
    process_custom_json_to_landmarks(VIDEO_DIR, OUTPUT_DIR, JSON_PATH)