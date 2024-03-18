import torch

# 假设 combined_tensor 是您组合后的张量
combined_tensor = torch.tensor([[[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]], [[4.0, 0.4], [5.0, 0.5], [6.0, 0.6]]])

# 拆分回原始的两个张量
first_point_mu = combined_tensor[:, :, 0]
first_point_std = combined_tensor[:, :, 1]

print("first_point_mu:")
print(first_point_mu)
print("first_point_std:")
print(first_point_std)
