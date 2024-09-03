import os
import re

import numpy as np
import torch
from stage1.dataset import normalize_pose, get_cluster_dataset
from stage1.fusion import Model
from stage1.knowledge_graph import search_relation, create_relation
from tqdm import tqdm


def txt2array():
    file_name = "/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/cluster_centers.txt"
    # 读取txt文件
    with open(file_name, 'r') as file:
        lines = file.readlines()

    # 解析每一行并存储为数组
    data_arrays = []
    for line in lines:
        # 使用空格分割每一行的数据
        values = line.split()[1:]

        # 将字符串转换为浮点数
        values = [float(value) for value in values]

        # 将每一行的数组添加到总数组中
        data_arrays.append(values)
    return data_arrays

def cal_similarity(query,arrays):
    query_array = query
    data_arrays = arrays
    cosine_similarity_max = -1
    index = -1
    for i in range(len(data_arrays)):
        data_array = data_arrays[i]
        cosine_similarity = np.dot(query_array, data_array) / (np.linalg.norm(query_array) * np.linalg.norm(data_array))
        if cosine_similarity_max < cosine_similarity:
            cosine_similarity_max = cosine_similarity
            index = i
    return index


def find_cluster_pose(pose,scene,relation):
    if relation == 'abnormal':
        if search_relation('scene', 'pose', scene, pose) != 'normal':
            create_relation('scene', 'pose', scene, pose, relation)
    elif relation == 'normal':
        create_relation('scene', 'pose', scene, pose, relation)



def cluster_test(auc_1=0,flag1=0):
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
    dataset_n, dataset_a, scene_n, scene_a = get_cluster_dataset()

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

    array = txt2array()
    pose_index_n = []
    pose_index_a = []
    num = 0
    index_old = -1
    with tqdm(total=len(pose_n),desc="知识图谱关系(normal)创建中") as pbar:
        for i in range(len(pose_n)):
            index = 1+cal_similarity(pose_n[i],array)
            if index_old == index:
                num+=1
            else:
                num = 1
                index_old = index
            if num == 5:
                find_cluster_pose(f'pose{index_old}', f'scene{scene_n[i]}', 'normal')
                # print(f'pose{index_old}\tscene{scene_n[i]}\tnormal')
                num = 0
                index_old = -1
            pbar.update(1)

    num = 0
    index_old = -1
    with tqdm(total=len(pose_a),desc="知识图谱关系(abnormal)创建中") as pbar:
        for i in range(len(pose_a)):
            index = 1+cal_similarity(pose_a[i],array)
            if index_old == index:
                num+=1
            else:
                num = 1
                index_old = index
            if num == 5:
                find_cluster_pose(f'pose{index_old}', f'scene{scene_a[i]}', 'abnormal')
                # print(f'pose{index_old}\tscene{scene_a[i]}\tabnormal')
                num = 0
                index_old = -1
            pbar.update(1)


if __name__ == '__main__':
    cluster_test()