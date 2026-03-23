import os
import sys

# 先打印，让你知道程序没死
print(">>> 正在初始化系统，请稍候（可能需要 1 分钟）...")

try:
    import cv2
    import numpy as np
    from collections import deque
    print(">>> 基础库加载完毕...")
    
    import torch
    print(f">>> PyTorch 版本: {torch.__version__} 加载完毕...")
    
    from ultralytics import YOLO
    print(">>> YOLO 模型组件加载完毕...")
except Exception as e:
    print(f"加载库时出错: {e}")
    sys.exit()

# ==========================================
# 1. 初始化模型
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f">>> 最终运行设备: {device}")

# 加载模型（此处可能还会有一小段 CUDA 初始化时间）
pose_model = YOLO('yolo11n-pose.pt').to(device)

class SignLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SignLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

NUM_CLASSES = 93 
lstm_model = SignLSTM(34, 128, 2, NUM_CLASSES).to(device)

if os.path.exists('sign_language_lstm_v1.pth'):
    lstm_model.load_state_dict(torch.load('sign_language_lstm_v1.pth', map_location=device))
    lstm_model.eval()
    print(">>> 所有权重加载成功！")
else:
    print("!!! 错误：找不到 sign_language_lstm_v1.pth")
    sys.exit()

# ==========================================
# 2. 启动摄像头
# ==========================================
cap = cv2.VideoCapture(0)
pts_queue = deque(maxlen=60)
current_prediction = "Waiting..."

print("\n--- 系统已就绪！现在开始捕获摄像头 ---")
print("提示：请确保你在光线充足的地方，按 'q' 键退出。")

with torch.no_grad():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        results = pose_model(frame, verbose=False, conf=0.5)
        
        if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xyn) > 0:
            points = results[0].keypoints.xyn[0].cpu().numpy()
            pts_queue.append(points.flatten())
        else:
            pts_queue.append(np.zeros(34))
            
        if len(pts_queue) == 60:
            input_data = torch.FloatTensor(np.array(pts_queue)).unsqueeze(0).to(device)
            output = lstm_model(input_data)
            prob = torch.nn.functional.softmax(output, dim=1)
            conf, pred_idx = torch.max(prob, 1)
            
            if conf.item() > 0.4:
                current_prediction = f"ID: {pred_idx.item()} Conf: {conf.item():.2f}"
        
        cv2.rectangle(frame, (0, 0), (450, 80), (0, 0, 0), -1)
        cv2.putText(frame, current_prediction, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Translator', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()