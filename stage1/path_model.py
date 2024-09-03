import numpy as np
import torch
import torch.nn as nn

class PathFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PathFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM 层
        output, (hidden, cell) = self.lstm(x)
        # 只使用最后一个时间步的隐藏状态
        last_hidden = hidden[-1]
        # 全连接层
        features = self.linear(last_hidden)
        return features

# 定义模型参数
input_size = 2 # 每帧的数据维度 (x, y)
hidden_size = 50 # LSTM 隐藏层大小
output_size = 20 # 输出特征向量的大小

# 创建模型实例
model = PathFeatureExtractor(input_size, hidden_size, output_size)

# 打印模型结构
print(model)

skeleton_feature = torch.Tensor(np.random.rand(256, 24, 2))
feature = model(skeleton_feature)
print(feature.shape)
