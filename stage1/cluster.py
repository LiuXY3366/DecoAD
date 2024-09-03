import os
import re

import numpy as np
import torch
from scipy.spatial import distance
from stage1.dataset import get_cluster_dataset
from stage1.fusion import Model
from tqdm import tqdm

from stage1.knowledge_graph import create_relation


def read_first_line_from_file(filename):
    with open(filename, 'r') as file:
        first_line = file.readline().strip()  # 读取第一行并去除两端的空白字符

    parts = first_line.split()  # 用空白字符分割字符串
    name = parts[0]  # 第一个元素是名称
    values = list(map(float, parts[1:]))  # 将剩余部分转换为浮点数列表

    return name, values

def read_file_process_lines(filename):
    data = []  # 用于存储所有的数据
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()  # 分割每行并去除空白
            if len(parts) > 1:  # 确保行中有足够的数据
                name = parts[0]  # 第一个元素是名称
                # print(parts[1:])
                values = list(map(float, parts[1:]))  # 将剩余部分转换为浮点数列表
                data.append((name, values))  # 将名称和数值列表以元组形式添加到数据列表中

    return data


def cal_sim(file_name,pose_feature,num,state='N'):
    max_sim = -1.
    name_flag = ''
    # 打开文件以写入数据
    with open(file_name, "r"):
        data = read_file_process_lines(file_name)
        for name, values in data:
            cosine_similarity = 1 - distance.cosine(values, pose_feature)
            if cosine_similarity > max_sim:
                max_sim = cosine_similarity
                name_flag = name
                # print("余弦相似度：", cosine_similarity)
        if max_sim<0.4:
            with open(file_name, "a") as file:
                # 将标签和聚类中心转换为字符串格式
                label_str = f'{state}pose{num+1}'
                center_str = " ".join(map(str, pose_feature))  # 将聚类中心的每个元素转换为字符串并用空格分隔
                # 将标签和聚类中心写入文件
                file.write(f"{label_str} {center_str}\n")
                num+=1
    return name_flag,num

def find_cluster_pose(pose,scene,relation):
    'pose:pose6  scene:scene9  relation:normal'
    create_relation('scene', 'pose', scene, pose, relation)
    # if relation == 'abnormal':
    #     if search_relation('scene', 'pose', scene, pose) != 'normal':
    #         create_relation('scene', 'pose', scene, pose, relation)
    # elif relation == 'normal':
    #     create_relation('scene', 'pose', scene, pose, relation)

def get_cluster(auc_1 = 0.,flag1 = 0):
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
        dataset_n, dataset_a,scene_n,scene_a = get_cluster_dataset()

        with tqdm(total=len(dataset_n)) as pbar:
            for data, scene in zip(dataset_n, scene_n):
                # print(scene)
                # 使用模型进行推理
                pose_input = torch.from_numpy(data).to(torch.float).to(device)
                with torch.no_grad():
                    pose_feature = model(pose_input.unsqueeze(0)).squeeze(0).cpu().detach().numpy()
                    if not os.path.exists(file_name):
                        with open(file_name, "w") as file:
                            # 将标签和聚类中心转换为字符串格式
                            label_str = f'Npose1'
                            num = 1
                            center_str = " ".join(map(str, pose_feature))  # 将聚类中心的每个元素转换为字符串并用空格分隔
                            # 将标签和聚类中心写入文件
                            file.write(f"{label_str} {center_str}\n")
                    name_flag,num = cal_sim(file_name,pose_feature,num)
                    if name_flag.split('pose')[0] == 'N':
                        relation = 'normal'
                    'pose:pose6  scene:scene9  relation:normal'
                    # print(f"pose:pose{name_flag.split('pose')[1]}  scene:scene{scene}  relation:{relation}")
                    find_cluster_pose(f"pose{name_flag.split('pose')[1]}",f'scene{scene}',relation)
                    # print(name_flag)  # Npose8     Apose32
                    # print(f'pose_feature:{pose_feature}')
                    '''
                    pose_feature:[-4.845054    1.5303189   1.5376606   5.071626    0.61892563 -0.30140257]
                    '''
                pose_n.append(pose_feature)
                pbar.update(1)
        flag = False
        with tqdm(total=len(dataset_a)) as pbar:
            for data, scene in zip(dataset_a, scene_a):
                # 使用模型进行推理
                pose_input = torch.from_numpy(data).to(torch.float).to(device)
                with torch.no_grad():
                    pose_feature = model(pose_input.unsqueeze(0)).squeeze(0).cpu().detach().numpy()
                    if flag == False:
                        num+=1
                        flag = True
                        with open(file_name, "a") as file:
                            # 将标签和聚类中心转换为字符串格式
                            label_str = f'Apose{num}'
                            center_str = " ".join(map(str, pose_feature))  # 将聚类中心的每个元素转换为字符串并用空格分隔
                            # 将标签和聚类中心写入文件
                            file.write(f"{label_str} {center_str}\n")
                            # return
                    name_flag,num = cal_sim(file_name,pose_feature,num,'A')
                    if name_flag.split('pose')[0] == 'N':
                        relation = 'normal'
                    elif name_flag.split('pose')[0] == 'A':
                        relation = 'abnormal'
                    'pose:pose6  scene:scene9  relation:normal'
                    # print(f"pose:pose{name_flag.split('pose')[1]}  scene:scene{scene}  relation:{relation}")
                    find_cluster_pose(f"pose{name_flag.split('pose')[1]}",f'scene{scene}',relation)
                    # print(name_flag)
                pose_a.append(pose_feature)
                pbar.update(1)

def pose_cluster(auc_1 = 0.,flag1 = 0):
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
        dataset_n, dataset_a,scene_n,scene_a = get_cluster_dataset()
        dataset_na = dataset_n+dataset_a
        scene_na = scene_n + scene_a

        nnn = 0
        with tqdm(total=len(dataset_na)) as pbar:
            for data, scene in zip(dataset_na, scene_na):
                # print(scene)
                # 使用模型进行推理
                pose_input = torch.from_numpy(data).to(torch.float).to(device)
                with torch.no_grad():
                    pose_feature = model(pose_input.unsqueeze(0)).squeeze(0).cpu().detach().numpy()
                    if not os.path.exists(file_name):
                        with open(file_name, "w") as file:
                            # 将标签和聚类中心转换为字符串格式
                            label_str = f'Npose1'
                            num = 1
                            center_str = " ".join(map(str, pose_feature))  # 将聚类中心的每个元素转换为字符串并用空格分隔
                            # 将标签和聚类中心写入文件
                            file.write(f"{label_str} {center_str}\n")
                    name_flag,num = cal_sim(file_name,pose_feature,num)
                    nnn = num
                    if name_flag.split('pose')[0] == 'N':
                        relation = 'normal'
                    'pose:pose6  scene:scene9  relation:normal'
                    # print(f"pose:pose{name_flag.split('pose')[1]}  scene:scene{scene}  relation:{relation}")
                    find_cluster_pose(f"pose{name_flag.split('pose')[1]}",f'scene{scene}',relation)
                    # print(name_flag)  # Npose8     Apose32
                    # print(f'pose_feature:{pose_feature}')
                    '''
                    pose_feature:[-4.845054    1.5303189   1.5376606   5.071626    0.61892563 -0.30140257]
                    '''
                pose_n.append(pose_feature)
                pbar.update(1)
        print(f'nnn:{nnn}')

def scene_cluster(auc_1 = 0.,flag1 = 0):
    # 指定要保存的文件名
    file_name = "/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/scene_cluster_centers.txt"
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
        dataset_na = []
        # dataset_n, dataset_a,scene_n,scene_a = get_cluster_dataset()
        for i in range(29):
            data = torch.load(f'/home/liuxinyu/PycharmProjects/KG-VAD/data/UBnormal_scene_feature/scene{i+1}_features.pth')
            dataset_na+=data
        # scene_na = scene_n + scene_a

        nnn = 0
        with tqdm(total=len(dataset_na)) as pbar:
            for data in dataset_na:
                # print(scene)
                # 使用模型进行推理
                # pose_input = torch.from_numpy(data).to(torch.float).to(device)
                pose_input = torch.from_numpy(data.numpy()).to(torch.float).to(device)
                with torch.no_grad():
                    pose_feature = pose_input.tolist()
                    # print(pose_feature)
                    if not os.path.exists(file_name):
                        with open(file_name, "w") as file:
                            # 将标签和聚类中心转换为字符串格式
                            label_str = f'Npose1'
                            num = 1
                            center_str = " ".join(map(str, pose_feature))  # 将聚类中心的每个元素转换为字符串并用空格分隔
                            # 将标签和聚类中心写入文件
                            file.write(f"{label_str} {center_str}\n")
                    name_flag,num = cal_sim(file_name,pose_feature,num)
                    nnn = num
                    if name_flag.split('pose')[0] == 'N':
                        relation = 'normal'
                    'pose:pose6  scene:scene9  relation:normal'
                    # print(f"pose:pose{name_flag.split('pose')[1]}  scene:scene{scene}  relation:{relation}")
                    # find_cluster_pose(f"pose{name_flag.split('pose')[1]}",f'scene{scene}',relation)
                    # print(name_flag)  # Npose8     Apose32
                    # print(f'pose_feature:{pose_feature}')
                    '''
                    pose_feature:[-4.845054    1.5303189   1.5376606   5.071626    0.61892563 -0.30140257]
                    '''
                pose_n.append(pose_feature)
                pbar.update(1)
        print(f'nnn:{nnn}')

def main():
    auc_1 = 0.78500
    flag1 = 1
    get_cluster(auc_1,flag1)


if __name__ == '__main__':
    main()