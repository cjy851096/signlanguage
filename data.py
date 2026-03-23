import cv2
import os
import numpy as np
import json
from ultralytics import YOLO
from tqdm import tqdm
import torch

# ==========================================
# 1. 环境与模型初始化
# ==========================================
# 检查是否可用 GPU，如果可用则使用 'cuda'，否则使用 'cpu'
# 注意：YOLO 的 .to() 也可以直接接受 'cuda' 字符串
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# 加载 YOLO11-Pose 模型
model = YOLO('yolo11n-pose.pt').to(device)


# ==========================================
# 2. 特征提取函数定义 (已修复 Json 解析 Bug)
# ==========================================
def process_wlasl_to_landmarks(video_dir, output_dir, json_path):
    # 加载标签数据
    with open(json_path, 'r') as f:
        wlasl_data = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 关键点修复：针对不同版本的 JSON 结构进行兼容
    # 逻辑：如果 wlasl_data 是字典，取其 values；如果是列表，直接使用
    if isinstance(wlasl_data, dict):
        items_to_process = list(wlasl_data.values())
    else:
        items_to_process = wlasl_data

    print(f"Total entries found in JSON: {len(items_to_process)}")

    # 遍历视频
    for entry in tqdm(items_to_process):
        # 防御性编程：确保 entry 是字典格式
        if not isinstance(entry, dict):
            continue
            
        video_id = str(entry.get('video_id'))
        label = entry.get('label')
        
        if not video_id or not label:
            continue

        # 拼接视频完整路径 (WLASL 视频通常为 .mp4)
        video_path = os.path.join(video_dir, f"{video_id}.mp4")

        # 如果找不到视频文件则跳过
        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        sequence_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break

            # 使用 YOLO 进行姿态推理
            # verbose=False 禁用每一帧的打印，提高速度
            results = model(frame, verbose=False, conf=0.5)
            
            if len(results) > 0 and results[0].keypoints is not None:
                # 提取归一化后的 [17, 2] 坐标 (x, y)
                # xyn 属性已经是 0-1 之间的浮点数
                points = results[0].keypoints.xyn[0].cpu().numpy() 
                # 如果检测到的点全为 0，说明该帧没识别到人
                sequence_data.append(points)
            else:
                # 填充零矩阵，保持时序长度一致
                sequence_data.append(np.zeros((17, 2)))

        cap.release()

        # 保存为 .npy 文件，文件名包含 ID 和 标签方便后续训练读取
        if len(sequence_data) > 0:
            save_path = os.path.join(output_dir, f"{video_id}_{label}.npy")
            np.save(save_path, np.array(sequence_data))

# ==========================================
# 3. 路径配置与执行转换
# ==========================================
# 请根据你 Kaggle 左侧 Data 栏的具体路径修改以下三行
# 提示：点击文件夹旁边的三个点选择 "Copy path"
VIDEO_DIR = r'E:\signlanguage\archive\videos' 
JSON_PATH = r'E:\signlanguage\archive\nslt_100.json'
OUTPUT_DIR = r'E:\signlanguage\output'

print("--- Starting Feature Extraction ---")
if not os.path.exists(VIDEO_DIR):
    print(f"Error: Video directory not found at {VIDEO_DIR}")
    # 打印输入目录内容帮助调试
else:
    process_wlasl_to_landmarks(VIDEO_DIR, OUTPUT_DIR, JSON_PATH)

    # 4. 最终检查
    if os.path.exists(OUTPUT_DIR):
        generated_files = os.listdir(OUTPUT_DIR)
        print(f"--- Process Finished ---")
        print(f"Successfully extracted features for {len(generated_files)} videos.")
        if len(generated_files) > 0:
            print(f"Sample file: {generated_files[0]}")