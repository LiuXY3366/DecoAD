import time

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


torch.set_default_tensor_type('torch.cuda.FloatTensor')


def get_adjacency_matrices():
    N = 18
    adjacency_matrix = np.zeros((N, N), dtype=np.int64)
    neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                     (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                     (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
    edge_index = []
    for link in neighbor_link:
        i, j = link
        # adjacency_matrix[i][j] = 1  # Set the connection from i to j
        edge_index.append([i, j])
    # adjacency_matrix is now your adjacency matrix
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# 数据处理函数，将输入数据调整为 GCN 的输入格式
def preprocess_data(data):
    n, num_frames, _, num_nodes = data.shape
    data = data.reshape(n * num_frames, -1, num_nodes).permute(0, 2, 1).contiguous()
    return data

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # print(x.shape)   # torch.Size([3072, 2, 18])
        # print(edge_index.shape)  # (18, 18)
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, pose):
        batch_size, seq_len, input_size = pose.size()
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        # 重塑输入数据形状以适应LSTM
        pose = pose.view(batch_size, seq_len, input_size)
        # LSTM 前向传播
        pose, (h_n, c_n) = self.lstm(pose, (h_0, c_0))
        # 选择最后一个时间步的输出
        # pose = pose[:, -1, :]  # 选择最后一个时间步的隐藏状态作为输出
        return pose

class GetPose(nn.Module):
    def __init__(self):
        super(GetPose, self).__init__()
        self.adj_matrix = get_adjacency_matrices()
        self.gcn = GCN(in_channels=2, hidden_channels=16, out_channels=2)
        self.lstm = LSTM(input_size=36,hidden_size=128,num_layers=1)
        self.conv = nn.Conv1d(in_channels=24, out_channels=1, kernel_size=3, padding=1)

    def forward(self, pose):
        pose = pose.permute(0, 2, 1, 3)
        pose = self.gcn(preprocess_data(pose), self.adj_matrix)
        pose = pose.reshape(-1,24,36)
        pose = self.lstm(pose)
        pose = self.conv(pose).squeeze(1)
        return pose

class PathFeatureExtractor(nn.Module):
    '''
    hidden_size = 16, output_size = 4      auc:0.6653157513519865
    hidden_size = 32, output_size = 16     auc:0.5028051091642114
    hidden_size = 32, output_size = 8      auc:0.4990213716051083
    '''
    def __init__(self, input_size = 2, hidden_size = 32, output_size = 4):
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



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.getpose = GetPose()
        self.pfe = PathFeatureExtractor()
        self.classifier = nn.Sequential(
            nn.Linear(512+4+128,256),
            nn.ReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self, pose,path = '',scene=''):
        # print(scene.shape)
        # start_time = time.perf_counter()
        pose = self.getpose(pose)
        # cost_time = time.perf_counter() - start_time
        # print(f'pose:{np.sum(cost_time)}')  # 3691.3083049989655
        if scene != '':
            # start_time = time.perf_counter()
            pfe = self.pfe(path)  # torch.Size([256, 20])
            # cost_time = time.perf_counter() - start_time
            # print(f'pfe:{np.sum(cost_time)}')
            # start_time = time.perf_counter()
            scene = scene.squeeze(1).squeeze(1)
            # cost_time = time.perf_counter() - start_time
            # print(f'scene:{np.sum(cost_time)}')
            # scene = self.conv1d(scene)
            # start_time = time.perf_counter()
            scene = scene.squeeze(1)
            scene = torch.cat((pfe,scene),dim=1)
            fusion = torch.cat((pose,scene),dim=1)

            x = self.classifier(fusion)
            # cost_time = time.perf_counter() - start_time
            # print(f'fusion:{np.sum(cost_time)}')
            return x
        else:
            return pose

# start_time = time.perf_counter()
# cost_time = time.perf_counter() - start_time
# print(f'pose:{np.sum(cost_time)}')
'''
dataset
    - split
        - train
        - test
    - data 有n个下述属性
        - data_np   (3,24,18)  # ((x坐标、y坐标、置信度),24帧,18个像素点）
        - trans_index   # 是否进行了数据增强，进行了何种数据增强，默认0。在这好像也是0
        - seg_score  # 整个视频帧的置信分数
        - label   # 标签  train:1是正常（4700） -1是异常（142058）   test:1应该是不考虑是否异常（301034）

    data:[
            [data_np,trans_index,seg_score,label],
            [data_np,trans_index,seg_score,label],
            [data_np,trans_index,seg_score,label],
            [data_np,trans_index,seg_score,label],
            ... ...
         ]
'''





def main():
    # 假设骨骼视频特征形状为 (256, 24, 36) 和场景特征形状为 (256, 1000)
    skeleton_feature = torch.Tensor(np.random.rand(128,2,24,18))
    scene_feature = torch.Tensor(np.random.rand(128,1, 512))
    path_feature = torch.Tensor(np.random.rand(128, 24, 2))

    # 创建模型实例
    model = Model()
    # model.apply(weight_init)

    # 进行融合处理
    output = model(skeleton_feature,path_feature,scene_feature)
    print(output.shape)   # torch.Size([1, 24, 36])
    print(output)   # torch.Size([1, 24, 36])

if __name__ == '__main__':
    main()
    # gen_fusion_dataset()