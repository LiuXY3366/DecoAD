import os
import re
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from UNVAD.stage1.args import init_parser, init_sub_args
from UNVAD.stage1.dataset import gen_fusion_dataset_dataloader, get_dataset_and_loader, trans_list, get_cluster_dataset
from UNVAD.stage1.fusion import Autoencoder
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from UNVAD.stage1.args import init_parser

# 初始化解析器
parser = init_parser()

# 解析参数
args = parser.parse_args()


def save2txt(cluster_info,filename,category = 'pose'):
    # 指定要保存的文件名
    file_name = filename

    # 检查文件是否已存在
    if not os.path.exists(file_name):
        # 如果文件不存在，执行以下操作

        # 打开文件以写入数据
        with open(file_name, "w") as file:
            # 遍历 cluster_info 列表
            for i in range(len(cluster_info)):
                # 将标签和聚类中心转换为字符串格式
                label_str = f'{category}{i+1}'
                center_str = " ".join(map(str, cluster_info[i]))  # 将聚类中心的每个元素转换为字符串并用空格分隔

                # 将标签和聚类中心写入文件
                file.write(f"{label_str} {center_str}\n")
    else:
        print(f"File '{file_name}' already exists. Skipping the saving process.")

def get_all_npy_paths(root_folder):
    npy_paths = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.pth'):
                full_path = os.path.join(root, file)
                npy_paths.append(full_path)
    all_data = []
    for path in npy_paths:
        # print(path)
        data = torch.load(path).expand(1,512)
        all_data.append(data)
    return np.concatenate(all_data, axis=0)  # 正确返回整合后的 numpy 数组

def get_scene_cluster(num = 40):
    # 指定要保存的文件名
    file_name = f"{args.UNVAD}KG/data/cluster/cluster_scene_centers.txt"
    # 检查文件是否存在
    if os.path.exists(file_name):
        os.remove(file_name)
    if not os.path.exists(file_name):
        device = torch.device("cuda:0")
        scene_inputs = torch.from_numpy(get_all_npy_paths(f'{args.DATA}UFSR_scene_feature')).to(torch.float).to(device)

        kmeans = KMeans(n_clusters=num, random_state=0).fit(np.array(scene_inputs.cpu()))

        # 获取聚类中心
        centers = kmeans.cluster_centers_

        '''
        matched：[ True  True  True  True  True False  True  True  True False False False
                   True  True  True  True  True  True  True  True]
        '''
        centers = list(centers)  # Convert centers2 to a Python list

        # print(f'cluster2:{centers}')

        print(f'cluster2:{len(centers)}')

        save2txt(centers, file_name,category='scene')

        return len(centers)

def get_cluster(auc_1,flag1,threshold1,dataset_input=None,loader_input=None,batchsize = args.opbs):
    # 指定要保存的文件名
    file_name = f"{args.UNVAD}KG/data/cluster/cluster_centers.txt"
    # 检查文件是否存在
    if os.path.exists(file_name):
        os.remove(file_name)
    if not os.path.exists(file_name):
        name = f'{args.supervise}|{args.dataset}'
        device = torch.device("cuda:0")
        model = Autoencoder()

        # 加载预训练的权重
        checkpoints = f'{args.UNVAD}stage{flag1}/ckpt/{name}/model' + '{:.5f}'.format(
            auc_1) + '_' + '{:.5f}.pkl'.format(
            threshold1)
        # print("Keys in the checkpoint:", checkpoint.keys())

        checkpoint = torch.load(checkpoints)

        # 仅加载 getpose 部分的权重
        getpose_state_dict = {
            key.replace('encoder.getpose.', ''): value
            for key, value in checkpoint.items() if key.startswith('encoder.getpose.')
        }
        model.encoder.getpose.load_state_dict(getpose_state_dict)

        # 设置模型为评估模式
        model.eval()

        # 现在 pose_feature_output 包含了模型的 pose 特征输出
        pose_n = []
        # pose_a = []
        dataset_n, _,_,_,dataset_input,loader_input = get_cluster_dataset(dataset_input,loader_input)

        # 确保 dataset_n 是一个 NumPy 数组
        dataset_n = np.array(dataset_n)  # 转换成 NumPy 数组

        # 329170
        pose_inputs = torch.from_numpy(dataset_n).to(torch.float).to(device)
        # 使用 DataLoader 处理批次数据
        for start_idx in range(0, len(pose_inputs), batchsize):
            # 动态计算结束索引，确保不会超出数组范围
            end_idx = min(start_idx + batchsize, len(pose_inputs))
            batch_data = pose_inputs[start_idx:end_idx]  # 取出当前批次的数据

            # 使用模型进行推理
            with torch.no_grad():
                pose_features = model(batch_data)  # 批量推理

            # 将每个样本的特征展平成一维并添加到列表中
            pose_n.extend(pose_features.cpu().detach().numpy().reshape(len(batch_data), -1))

        # 现在 pose_n 是一个包含所有特征的列表，转换为 NumPy 数组进行 KMeans 聚类
        pose_n = np.array(pose_n)  # 变成 NumPy 数组

        # 执行 KMeans 聚类
        kmeans1 = KMeans(n_clusters=25,n_init=10, random_state=0).fit(pose_n)
        # print(kmeans1)

        # 获取聚类中心
        centers1 = kmeans1.cluster_centers_

        centers1 = list(centers1)  # Convert centers2 to a Python list


        print(f'cluster2:{centers1}')

        print(f'cluster2:{len(centers1)}')

        save2txt(centers1, file_name)

        return len(centers1),dataset_input,loader_input

def main():
    get_cluster()


if __name__ == '__main__':
    main()
