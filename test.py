import numpy as np

# 示例数据
targets = [np.array([1, 2, 3]), np.array([4, 5, 6])]
preds = [np.array([1.1, 2.1, 3.1]), np.array([4.1, 5.1, 6.1])]
labels = [np.array([0, 1, 1]), np.array([1, 0, 0])]

# 将多个数组沿轴0拼接
targets = np.concatenate(targets, axis=0)
preds = np.concatenate(preds, axis=0)
labels = np.concatenate(labels, axis=0)

print("Targets:", targets)
print("Preds:", preds)
print("Labels:", labels)
