import os
import re
import numpy as np
import torch
from sklearn.cluster import KMeans

from stage1.args import init_parser, init_sub_args
from stage1.dataset import gen_fusion_dataset_dataloader, get_dataset_and_loader, trans_list, get_cluster_dataset
from stage1.fusion import Model
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def save2txt(cluster_info,filename):
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
                label_str = f'pose{i+1}'
                center_str = " ".join(map(str, cluster_info[i]))  # 将聚类中心的每个元素转换为字符串并用空格分隔

                # 将标签和聚类中心写入文件
                file.write(f"{label_str} {center_str}\n")
    else:
        print(f"File '{file_name}' already exists. Skipping the saving process.")

def get_cluster(auc_1 = 0,flag1 = 0):
    # 指定要保存的文件名
    file_name = "/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/cluster_centers.txt"
    # 检查文件是否存在
    if os.path.exists(file_name):
        os.remove(file_name)
    if not os.path.exists(file_name):
        checkpoints = f'/home/liuxinyu/PycharmProjects/KG-VAD/stage{flag1}/ckpt/model'+'{:.5f}.pkl'.format(auc_1)
        device = torch.device("cuda:0")
        model = Model()

        # 加载预训练的权重
        checkpoint = torch.load(checkpoints)
        print("Keys in the checkpoint:", checkpoint.keys())

        # 仅加载 getpose 部分的权重
        getpose_state_dict = {
            key.replace('getpose.', ''): value
            for key, value in checkpoint.items() if key.startswith('getpose.')
        }
        model.getpose.load_state_dict(getpose_state_dict)

        # 设置模型为评估模式
        model.eval()

        # 现在 pose_feature_output 包含了模型的 pose 特征输出
        pose_n = []
        pose_a = []
        dataset_n, dataset_a,_,_ = get_cluster_dataset()

        for data in dataset_n:
            # 使用模型进行推理
            pose_input = torch.from_numpy(data).to(torch.float).to(device)
            with torch.no_grad():
                pose_feature = model(pose_input.unsqueeze(0)).squeeze(0).cpu().detach().numpy()
            pose_n.append(pose_feature)

        for data in dataset_a:
            # 使用模型进行推理
            pose_input = torch.from_numpy(data).to(torch.float).to(device)
            with torch.no_grad():
                pose_feature = model(pose_input.unsqueeze(0)).squeeze(0).cpu().detach().numpy()
            pose_a.append(pose_feature)

        kmeans1 = KMeans(n_clusters=15, random_state=0).fit(np.array(pose_n))
        kmeans2 = KMeans(n_clusters=25, random_state=0).fit(np.array(pose_a))

        # 获取聚类标签
        labels1 = kmeans1.labels_
        labels2 = kmeans2.labels_

        # 获取聚类中心
        centers1 = kmeans1.cluster_centers_
        centers2 = kmeans2.cluster_centers_

        # 初始化一个数组，用于存储第一组每个类别是否已被匹配
        matched = np.zeros(15, dtype=bool)

        # 初始化对应关系
        correspondence = np.arange(25)

        # 设置相似度阈值
        similarity_threshold = 0.94

        # 遍历第一组的每个类别
        for i in range(15):
            # 计算相似度
            similarities = 1 - cdist(centers1[i].reshape(1, -1), centers2, 'cosine')[0]

            # 找到相似度超过阈值的类别
            similar_indices = np.where(similarities >= similarity_threshold)[0]

            # 如果找到匹配的类别且该类别还未匹配
            if len(similar_indices) > 0 and not matched[i]:
                matched[i] = True
                correspondence[i] = similar_indices[0]

        # 现在，correspondence数组包含了第一组每个类别对应的第二组类别索引

        print(f'matched：{matched}')
        '''
        matched：[ True  True  True  True  True False  True  True  True False False False
                   True  True  True  True  True  True  True  True]
        '''
        centers2 = list(centers2)  # Convert centers2 to a Python list

        for i in range(centers1.shape[0]):
            if not matched[i]:
                centers2.append(centers1[i])

        print(f'cluster2:{centers2}')

        print(f'cluster2:{len(centers2)}')

        save2txt(centers2, file_name)

        return len(centers2)

def main():
    get_cluster()


if __name__ == '__main__':
    main()
