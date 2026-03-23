from collections import Counter

# 统计每个词汇出现的次数
counts = Counter(labels)

print("词汇样本统计 (前 20 个):")
for label, count in counts.most_common(93):
    print(f"{label}: {count} 个文件")