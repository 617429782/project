"""
# 这里存放一些超参数
"""
import torch

# 设置GPU环境，检查GPU是否可用
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print("device:")
print(device)
