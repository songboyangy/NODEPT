import numpy as np

def compute_derivative(values):
    derivatives = (values[:, 1:] - values[:, :-1])
    return derivatives



pred = np.array([[0.01534897, -0.00248897, -0.01526574, -0.00613825, 0.01806497, -0.00833355, 0.0589812, 0.13134463, 0.34043998, 0.5171923, 0.7708035, 1.0450442, 1.1293032]])
label = np.array([[0.0, 8.84235034, 10.05528244, 10.82575383, 11.49635449, 15.25779295, 15.41982854, 15.44795494, 15.46253422, 15.4807902, 15.5012152, 15.5416714, 16.532569]])

dy_pred_dt = compute_derivative(pred)
dy_true_dt = compute_derivative(label)

# 计算导数的损失
derivative_loss = np.mean(np.abs(dy_pred_dt - dy_true_dt))
print(derivative_loss)