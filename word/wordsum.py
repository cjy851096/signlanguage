import os

folder_path = '/kaggle/working/features_2000'

# 1. 获取所有 .npy 文件名
files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

# 2. 从文件名中提取 label (格式是 videoID_label.npy)
# split('_')[1] 获取 label，split('.')[0] 去掉后缀
labels = [f.split('_')[1].split('.')[0] for f in files]

# 3. 去重并排序
unique_labels = sorted(set(labels))

print(f"总计提取样本数: {len(files)}")
print(f"包含的唯一词汇数: {len(unique_labels)}")
print("具体词汇列表:")
print(unique_labels)