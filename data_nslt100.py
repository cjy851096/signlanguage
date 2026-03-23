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

# 2. 加载类别映射 (将数字 ID 转换为单词)
# 假设 wlasl_class_list.txt 格式为: 0    book
CLASS_LIST_PATH = '/kaggle/input/datasets/risangbaskoro/wlasl-processed/wlasl_class_list.txt'
idx_to_label = {}
with open(CLASS_LIST_PATH, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            idx_to_label[int(parts[0])] = parts[1]

def process_final(video_dir, output_dir, json_path):
    with open(json_path, 'r') as f:
        wlasl_data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)
    
    for video_id, info in tqdm(wlasl_data.items()):
        label_idx = info['action'][2]
        label = idx_to_label.get(label_idx, f"class_{label_idx}")
        
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        sequence_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            results = model(frame, verbose=False, conf=0.5)
            
            # --- 修复逻辑开始 ---
            # 必须同时满足：有结果、有关键点、且检测到的人数 > 0
            if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
                # 安全地提取第一个检测到的人
                points = results[0].keypoints.xyn[0].cpu().numpy() 
                sequence_data.append(points)
            else:
                # 如果没检测到人，填充 17 个零点
                sequence_data.append(np.zeros((17, 2)))
            # --- 修复逻辑结束 ---
            
        cap.release()

        if len(sequence_data) > 0:
            np.save(os.path.join(output_dir, f"{video_id}_{label}.npy"), np.array(sequence_data))
# 3. 设置路径并运行
VIDEO_DIR = '/kaggle/input/datasets/risangbaskoro/wlasl-processed/videos' 
JSON_PATH = '/kaggle/input/datasets/risangbaskoro/wlasl-processed/nslt_100.json'
OUTPUT_DIR = '/kaggle/working/features_2000'

process_final(VIDEO_DIR, OUTPUT_DIR, JSON_PATH)
print(f"提取完成！文件保存在: {OUTPUT_DIR}")