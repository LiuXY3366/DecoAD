import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


torch.set_default_tensor_type('torch.cuda.FloatTensor')


def get_adjacency_matrices():
    neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                     (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                     (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
    edge_index = []
    for link in neighbor_link:
        i, j = link
        edge_index.append([i, j])
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

# 修改后的GetPose类，去除了不必要的部分
class GetPose(nn.Module):
    def __init__(self):
        super(GetPose, self).__init__()
        self.adj_matrix = get_adjacency_matrices()
        self.gcn = GCN(in_channels=2, hidden_channels=24, out_channels=2)
        self.lstm = LSTM(input_size=36,hidden_size=128,num_layers=1)

    def forward(self, pose):
        pose = pose.permute(0, 2, 1, 3)
        pose = self.gcn(preprocess_data(pose), self.adj_matrix)
        pose = pose.reshape(-1,24,36)
        pose = self.lstm(pose)
        return pose

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.get_pose = GetPose()
        # 假设scene特征的维度是512（ResNet18的典型输出）
        self.fc = nn.Linear(128*24+512, 256)  # 128是LSTM的隐藏层大小

    def forward(self, pose, scene_feature):
        # 处理pose特征
        pose_feature = self.get_pose(pose).reshape(-1,128*24).unsqueeze(1)
        # print(pose_feature.size())  # torch.Size([128, 1, 2048])
        # print(scene_feature.size())  # torch.Size([128, 1, 512])
        # 连接pose和scene特征
        combined_feature = torch.cat((pose_feature, scene_feature), dim=2)
        # print(combined_feature.size())  # torch.Size([128, 1, 2560])
        # 使用全连接层进一步处理
        encoded_feature = F.relu(self.fc(combined_feature))
        # print(encoded_feature.size())
        return encoded_feature


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 1024)
        # self.fc0 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 36*24+512)
        self.relu = nn.ReLU()

    def forward(self, encoded_feature):
        x = self.relu(self.fc1(encoded_feature))
        x = self.relu(self.fc2(x))
        # x = self.relu(self.fc0(x))
        decoded_feature = self.fc3(x)
        return decoded_feature.squeeze(1)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, pose, scene_feature):
        encoded_feature = self.encoder(pose, scene_feature)
        # print(encoded_feature.size())
        reconstructed_pose = self.decoder(encoded_feature)
        return reconstructed_pose


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
    skeleton_feature = torch.Tensor(np.random.rand(2,2,24,18))
    scene_feature = torch.Tensor(np.random.rand(2,1, 512))

    batch_size = skeleton_feature.size(0)

    input = torch.cat((skeleton_feature.reshape(batch_size, -1), scene_feature.reshape(batch_size, -1)), dim=1)
    print(input.size())
    # 创建模型实例
    model = Autoencoder()
    # model.apply(weight_init)

    # 进行融合处理
    output = model(skeleton_feature,scene_feature)
    print(output.shape)   # torch.Size([128, 1, 128])
    # print(output)   # torch.Size([1, 24, 36])

if __name__ == '__main__':
    main()
    # gen_fusion_dataset()