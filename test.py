from sklearn.metrics import mean_squared_error
import numpy as np
import torch
# 模拟数据
pred = torch.tensor(np.array([[2.5, 3.5], [4.5, 5.5], [6.5, 7.5]]))
label = torch.tensor(np.array([[2, 3], [4, 5], [6, 7]]))

# 计算均方误差
mse = mean_squared_error(label, pred, multioutput='raw_values')
mse_mean=np.mean(mse)
print("均方误差数组:", mse)
print("整体的均方误差（MSLE）:", mse[0])
print(mse_mean)