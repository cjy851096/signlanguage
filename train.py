import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

# ==========================================
# 1. 数据集加载与对齐 (Padding)
# ==========================================
class SignLanguageDataset(Dataset):
    def __init__(self, feature_dir, max_len=60):
        self.feature_files = [f for f in os.listdir(feature_dir) if f.endswith('.npy')]
        self.feature_dir = feature_dir
        self.max_len = max_len
        
        # 自动提取所有类别并建立映射
        all_labels = [f.split('_')[1].replace('.npy', '') for f in self.feature_files]
        self.classes = sorted(list(set(all_labels)))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        print(f"检测到类别总数: {len(self.classes)}")

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.feature_dir, self.feature_files[idx])
        # 数据形状: [帧数, 17, 2] -> 展平为 [帧数, 34]
        data = np.load(file_path)
        data = data.reshape(data.shape[0], -1)
        
        # 时序对齐：统一长度为 max_len
        if len(data) < self.max_len:
            pad = np.zeros((self.max_len - len(data), data.shape[1]))
            data = np.vstack((data, pad))
        else:
            data = data[:self.max_len]
            
        label_name = self.feature_files[idx].split('_')[1].replace('.npy', '')
        label = self.class_to_idx[label_name]
        
        return torch.FloatTensor(data), torch.tensor(label)

# ==========================================
# 2. 定义 LSTM 识别模型
# ==========================================
class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SignLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [Batch, Seq_Len, Input_Size]
        out, _ = self.lstm(x)
        # 取最后一帧的隐藏状态作为整个序列的特征总结
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 3. 配置参数与启动训练
# ==========================================
# 初始化数据
dataset = SignLanguageDataset('/kaggle/working/features_2000', max_len=60)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SignLSTM(input_size=34, hidden_size=128, num_layers=2, num_classes=len(dataset.classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
print("开始训练语义识别模型...")
model.train()
for epoch in range(100):
    total_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {total_loss/len(train_loader):.4f}")

# 保存最终的语义识别模型
torch.save(model.state_dict(), 'sign_language_lstm_v1.pth')
print("训练完成！模型已保存为 sign_language_lstm_v1.pth")