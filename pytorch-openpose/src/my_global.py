"""
# 这里存放一些超参数
"""
import torch

# 设置GPU环境，检查GPU是否可用
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print("device:")
print(device)

# 视频数据集
# 每段视频提取的视频帧数
video_length = 15

#
batch_size = 4
numClasses = 6

# 原视频 + 增强方法 总数
trans_num = 3

# lstm相关参数
lstm_inputSize = 256
lstm_hiddenSize = 256
lstm_numLayers = 2 # 几层
lstm_directions = 1 # 1:单向； 2：双向
lstm_numSteps = 15 # numSteps = timeSteps
