import os
import re

import numpy as np
import torch
from WSVAD.stage1.dataset import normalize_pose, get_cluster_dataset
from WSVAD.stage1.fusion import Model
from WSVAD.KG.knowledge_graph import search_relation, create_relation
from tqdm import tqdm
from WSVAD.stage1.args import init_parser

# 初始化解析器
parser = init_parser()

# 解析参数
args = parser.parse_args()


def txt2array(category = 'pose'):
    if category == 'pose':
        file_name = f"{args.WSVAD}KG/data/cluster/cluster_centers.txt"
    else:
        file_name = f"{args.WSVAD}KG/data/cluster/cluster_scene_centers.txt"
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

def cal_similarity_batch(query_batch, arrays):
    """
    批量计算余弦相似度 (并行计算，利用 GPU)
    :param query_batch: 查询样本的 Tensor (batch_size, feature_dim)
    :param arrays: 数据集的 Tensor (dataset_size, feature_dim)
    :return: 每个查询样本与数据集的最大相似度索引
    """
    # 计算批量余弦相似度
    similarities = torch.matmul(query_batch, arrays.T)  # 批量矩阵乘法
    query_norm = torch.norm(query_batch, dim=1, keepdim=True)  # 查询样本的 L2 范数
    arrays_norm = torch.norm(arrays, dim=1, keepdim=True)  # 数据集的 L2 范数

    cosine_similarities = similarities / (query_norm * arrays_norm.T)  # 余弦相似度

    # 找到每个查询样本的最大相似度索引
    max_similarity_indices = torch.argmax(cosine_similarities, dim=1)

    return max_similarity_indices

def optimized_batch_calculation(pose_features, array):
    """
    优化后的批量相似度计算
    :param pose_features: 查询样本的特征 (batch_size, feature_dim)
    :param array: 数据集 (dataset_size, feature_dim)
    :return: 每个查询样本的相似度索引
    """
    pose_features_tensor = torch.stack([torch.tensor(feature).to('cuda') for feature in pose_features])
    array_tensor = torch.stack([torch.tensor(arr).to('cuda') for arr in array])

    # 计算每个 pose_feature 与整个数据集之间的相似度
    pose_ids_clip = cal_similarity_batch(pose_features_tensor, array_tensor)

    return pose_ids_clip.cpu().numpy()  # 返回的结果是 CPU 上的 NumPy 数组


def find_cluster_pose(pose,scene,relation):
    if relation == 'abnormal':
        if search_relation('scene', 'pose', scene, pose) != 'normal':
            create_relation('scene', 'pose', scene, pose, relation)
    elif relation == 'normal':
        create_relation('scene', 'pose', scene, pose, relation)



def cluster_test(auc_1=0,flag1=0,dataset_input=None,loader_input=None,batchsize = args.opbs):
    name = f'{args.supervise}|{args.dataset}'
    checkpoints = f'model' + '{:.5f}.pkl'.format(
        auc_1)
    device = torch.device("cuda:0")
    model = Model()

    # 加载预训练的权重
    checkpoint = torch.load(f'{args.WSVAD}stage{flag1}/ckpt/{name}/'+checkpoints)
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
    dataset_n, dataset_a, scene_n, scene_a, dataset_input, loader_input = get_cluster_dataset(dataset_input, loader_input)

    dataset_n = np.array(dataset_n)  # 转换成 NumPy 数组
    dataset_a = np.array(dataset_a)  # 转换成 NumPy 数组

    # 使用 DataLoader 处理批次数据
    pose_inputs = torch.from_numpy(dataset_n).to(torch.float).to(device)
    for start_idx in range(0, len(pose_inputs), batchsize):
        # 取出一个批次的数据，批次大小为 256
        end_idx = min(start_idx + batchsize, len(pose_inputs))
        batch_data = pose_inputs[start_idx:end_idx]  # 取出当前批次的数据
        # print(len(batch_data))

        # 使用模型进行推理
        with torch.no_grad():
            pose_features = model(batch_data)  # 批量推理

        # 将每个样本的特征展平成一维并添加到列表中
        pose_n.extend(pose_features.cpu().detach().numpy().reshape(len(batch_data), -1))

        # print(f"pose_n shape: {np.array(pose_n).shape}")

    # 现在 pose_n 是一个包含所有特征的列表，转换为 NumPy 数组进行 KMeans 聚类
    pose_n = np.array(pose_n)  # 变成 NumPy 数组

    # 使用 DataLoader 处理批次数据
    pose_inputs = torch.from_numpy(dataset_a).to(torch.float).to(device)
    for start_idx in range(0, len(pose_inputs), batchsize):
        # 取出一个批次的数据，批次大小为 256
        end_idx = min(start_idx + batchsize, len(pose_inputs))
        batch_data = pose_inputs[start_idx:end_idx]  # 取出当前批次的数据
        # print(len(batch_data))

        # 使用模型进行推理
        with torch.no_grad():
            pose_features = model(batch_data)  # 批量推理

        # 将每个样本的特征展平成一维并添加到列表中
        pose_a.extend(pose_features.cpu().detach().numpy().reshape(len(batch_data), -1))

        # print(f"pose_n shape: {np.array(pose_n).shape}")

    # 现在 pose_n 是一个包含所有特征的列表，转换为 NumPy 数组进行 KMeans 聚类
    pose_a = np.array(pose_a)  # 变成 NumPy 数组

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


def cluster_all_test(auc_1=0,flag1=0,dataset_input=None,loader_input=None,batchsize = args.opbs):
    name = f'{args.supervise}|{args.dataset}'
    checkpoints = f'model' + '{:.5f}.pkl'.format(
        auc_1)
    device = torch.device("cuda:0")
    model = Model()

    # 加载预训练的权重
    checkpoint = torch.load(f'{args.WSVAD}stage{flag1}/ckpt/{name}/'+checkpoints)
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
    s_n = []
    s_a = []
    dataset_n, dataset_a, scene_n, scene_a, dataset_input, loader_input = get_cluster_dataset(dataset_input, loader_input)

    # print(f'label_arr:{label_arr}')

    dataset_n = np.array(dataset_n)  # 转换成 NumPy 数组
    dataset_a = np.array(dataset_a)  # 转换成 NumPy 数组

    # 使用 DataLoader 处理批次数据
    pose_inputs = torch.from_numpy(dataset_n).to(torch.float).to(device)
    scene_inputs = scene_n
    batch_scene_data = []
    for start_idx in range(0, len(pose_inputs), batchsize):
        # 取出一个批次的数据，批次大小为 256
        end_idx = min(start_idx + batchsize, len(pose_inputs))
        batch_pose_data = pose_inputs[start_idx:end_idx]  # 取出当前批次的数据
        sd = scene_inputs[start_idx:end_idx]  # 取出当前批次的数据

        for i in range(len(sd)):
            if args.dataset == 'UFSR':
                # print(f'start_idx+i:{start_idx+i}')
                # if label_arr[start_idx+i] == 1:
                #     FN = 'N'
                # else:
                #     FN = 'F'
                batch_scene_data.append(torch.load(f"{args.data_dir}{args.dataset}_scene_feature/{sd[i].split('_')[0]}N/scene{sd[i]}_features.pth"))
            else:
                batch_scene_data.append(torch.load(f"{args.data_dir}{args.dataset}_scene_feature/{sd[i].split('_')[0]}/scene{sd[i]}_features.pth"))
        # print(len(batch_data))

        # 使用模型进行推理
        with torch.no_grad():
            pose_features = model(batch_pose_data)  # 批量推理

        # 将每个样本的特征展平成一维并添加到列表中
        pose_n.extend(pose_features.cpu().detach().numpy().reshape(len(batch_pose_data), -1))
        tensor_batch_scene_data = torch.stack(batch_scene_data)
        s_n.extend(tensor_batch_scene_data.cpu().detach().numpy().reshape(len(batch_scene_data), -1))

        # s_n.extend(torch.tensor(batch_scene_data).cpu().detach().numpy().reshape(len(batch_scene_data), -1))

        # print(f"pose_n shape: {np.array(pose_n).shape}")

    # 现在 pose_n 是一个包含所有特征的列表，转换为 NumPy 数组进行 KMeans 聚类
    pose_n = np.array(pose_n)  # 变成 NumPy 数组
    s_n = np.array(s_n)  # 变成 NumPy 数组
    # print(s_n)

    # 使用 DataLoader 处理批次数据
    pose_inputs = torch.from_numpy(dataset_a).to(torch.float).to(device)
    scene_inputs = scene_a
    batch_scene_data = []
    for start_idx in range(0, len(pose_inputs), batchsize):
        # 取出一个批次的数据，批次大小为 256
        end_idx = min(start_idx + batchsize, len(pose_inputs))
        batch_pose_data = pose_inputs[start_idx:end_idx]  # 取出当前批次的数据
        sd = scene_inputs[start_idx:end_idx]  # 取出当前批次的数据
        # print(len(batch_data))

        for i in range(len(sd)):
            if args.dataset == 'UFSR':
            #     if label_arr[start_idx+i] == 1:
            #         FN = 'N'
            #     else:
            #         FN = 'F'
                batch_scene_data.append(torch.load(
                    f"{args.data_dir}{args.dataset}_scene_feature/{sd[i].split('_')[0]}F/scene{sd[i]}_features.pth"))
            else:
                batch_scene_data.append(torch.load(
                    f"{args.data_dir}{args.dataset}_scene_feature/{sd[i].split('_')[0]}/scene{sd[i]}_features.pth"))

        # 使用模型进行推理
        with torch.no_grad():
            pose_features = model(batch_pose_data)  # 批量推理

        # 将每个样本的特征展平成一维并添加到列表中
        pose_a.extend(pose_features.cpu().detach().numpy().reshape(len(batch_pose_data), -1))
        tensor_batch_scene_data = torch.stack(batch_scene_data)
        s_a.extend(tensor_batch_scene_data.cpu().detach().numpy().reshape(len(batch_scene_data), -1))
        # s_a.extend(torch.tensor(batch_scene_data).cpu().detach().numpy().reshape(len(batch_scene_data), -1))

        # print(f"pose_n shape: {np.array(pose_n).shape}")

    # 现在 pose_n 是一个包含所有特征的列表，转换为 NumPy 数组进行 KMeans 聚类
    pose_a = np.array(pose_a)  # 变成 NumPy 数组
    s_a = np.array(s_a)  # 变成 NumPy 数组

    pose_array = txt2array('pose')
    scene_array = txt2array('scene')
    pose_index_n = []
    pose_index_a = []
    num = 0
    pose_old = -1
    scene_old = -1
    with tqdm(total=len(pose_n),desc="知识图谱关系(normal)创建中") as pbar:
        for i in range(len(pose_n)):
            pose_index = 1+cal_similarity(pose_n[i],pose_array)
            # print(f"s_n[i]:{s_n[i]}")
            # print("s_n[i] shape:", s_n[i].shape)
            # print("scene_array shape:", scene_array.shape)
            scene_index = 1+cal_similarity(s_n[i].reshape(-1),scene_array)
            if pose_old == pose_index:
                num+=1
            else:
                num = 1
                pose_old = pose_index
                scene_old = scene_index
            if num == 5:
                find_cluster_pose(f'pose{pose_old}', f'scene{scene_old}', 'normal')
                # print(f'pose{index_old}\tscene{scene_n[i]}\tnormal')
                num = 0
                pose_old = -1
                scene_old = -1
            pbar.update(1)

    num = 0
    pose_old = -1
    scene_old = -1
    with tqdm(total=len(pose_a),desc="知识图谱关系(abnormal)创建中") as pbar:
        for i in range(len(pose_a)):
            pose_index = 1+cal_similarity(pose_a[i],pose_array)
            scene_index = 1 + cal_similarity(s_a[i].reshape(-1), scene_array)
            if pose_old == pose_index:
                num+=1
            else:
                num = 1
                pose_old = pose_index
                scene_old = scene_index
            if num == 5:
                find_cluster_pose(f'pose{pose_old}', f'scene{scene_old}', 'abnormal')
                # print(f'pose{index_old}\tscene{scene_a[i]}\tabnormal')
                num = 0
                pose_old = -1
                scene_old = -1
            pbar.update(1)


if __name__ == '__main__':
    cluster_test()