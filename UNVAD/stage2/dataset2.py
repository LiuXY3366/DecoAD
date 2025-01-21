import math
import random
import copy

import numpy as np

from UNVAD.KG.cluster2kg import cal_similarity, txt2array, cal_similarity_batch, optimized_batch_calculation
from UNVAD.stage1.dataset import UbnormalDataset,gen_fusion_dataset_dataloader
import torch
from torch.utils.data import DataLoader

from UNVAD.stage1.fusion import Autoencoder
from UNVAD.KG.knowledge_graph import search_relation
from tqdm import tqdm
from UNVAD.stage1.args import init_parser
import torch.nn as nn

# 初始化解析器
parser = init_parser()

# 解析参数
args = parser.parse_args()

criterion = nn.MSELoss(reduction='none')

def get_scene_feature(file_path, scene_id):
    with open(file_path, 'r') as file:
        for line in file:
            # 按空格拆分每一行
            parts = line.strip().split()
            if len(parts) > 1 and parts[0] == scene_id:
                # 提取特征部分并转换为浮点数
                features = list(map(float, parts[1:]))
                return torch.tensor(features)

    raise ValueError(f"Scene ID '{scene_id}' not found in the file.")

'''
三个bag：
    纯正bag：在原正bag中存在的pose和scene使用stage1中的模型进行检测，异常分数大于0.8且在知识图谱中的关系是abnormal，重新拼凑的异常分数大于0.9
    正bag:不属于纯正bag和负bag
    负bag:原负bag，以及在原正bag中存在的pose和scene使用stage1中的模型进行检测，异常分数小于0.3且在知识图谱中的关系是normal，重新拼凑的异常分数小于0.1
损失函数：纯正bag和负bag进行二分类，正bag与负bag进行同一阶段相同
'''

# 0.0125
def gen_fusion_dataset_dataloader_2(auc_1 = 0,flag1=0,threshold1 = 0,batchsize = args.opbs):
    name = f'{args.supervise}|{args.dataset}'
    checkpoints = f'{args.UNVAD}stage{flag1}/ckpt/{name}/model' + '{:.5f}'.format(auc_1) + '_' + '{:.5f}.pkl'.format(
        threshold1)
    device = torch.device("cuda:0")
    model1 = Autoencoder()
    model2 = Autoencoder()

    # 加载预训练的权重
    checkpoint = torch.load(checkpoints)
    print("Keys in the checkpoint:", checkpoint.keys())

    # 仅加载 getpose 部分的权重
    getpose_state_dict = {
        key.replace('encoder.getpose.', ''): value
        for key, value in checkpoint.items() if key.startswith('encoder.getpose.')
    }


    model1.encoder.getpose.load_state_dict(getpose_state_dict)
    # 设置模型为评估模式
    model1.eval()
    model2.load_state_dict(checkpoint)
    model2.eval()

    datasetn, datasett, loader_n, loader_t = gen_fusion_dataset_dataloader()
    dataset_train = datasetn
    dataset_test = datasett
    dataset_a = []
    datasets = []
    dataset_t = dataset_test
    nn = 0
    aa = 0
    aplus = 0
    old_relation = ''
    old_scene_id = 0
    old_pose_id = 0
    loader_args = {'batch_size': 256, 'num_workers': 0, 'pin_memory': False}

    init = 0
    relation_map =  np.zeros((1000, 1000))
    array = txt2array('pose')
    pose_ids = np.zeros(len(dataset_train))
    scene_ids = np.zeros(len(dataset_train))
    pi = 0
    scores_dir = np.zeros(len(dataset_train))

    if args.dataset == 'UFSR':
        scene_array = txt2array('scene')
        for start_idx in tqdm(range(0, len(dataset_train), batchsize), desc="制作pose_ids中..."):
            batch_data = dataset_train[start_idx:start_idx + batchsize]
            data, mate, scene, label, path = zip(*batch_data)
            pose_input = torch.stack([torch.tensor(d, dtype=torch.float) for d in data]).to(device)
            scene_features = torch.stack([torch.tensor(s, dtype=torch.float) for s in scene]).reshape(len(batch_data), -1).to(device)

            with torch.no_grad():
                pose_features = model1(pose_input).reshape(len(batch_data), -1)
            pose_ids_clip = optimized_batch_calculation(pose_features, array) + 1
            scene_ids_clip = optimized_batch_calculation(scene_features, scene_array) + 1
            pose_ids[pi:pi + len(batch_data)] = pose_ids_clip  # 将加1后的值赋值到 pose_ids 对应位置
            scene_ids[pi:pi + len(batch_data)] = scene_ids_clip  # 将加1后的值赋值到 pose_ids 对应位置

            path_input = torch.stack([torch.tensor(p.reshape(24, 2), dtype=torch.float) for p in path]).to(device)
            scene_input = torch.stack([s.squeeze(0) for s in scene]).to(device)
            with torch.no_grad():
                data_ori = torch.cat((pose_input.reshape(len(batch_data), -1).to(device),
                                      scene_input.reshape(len(batch_data), -1).to(device)), dim=1)
                data_rec = model2(pose_input, path_input, scene_input)
                loss_recons = criterion(data_ori, data_rec).cpu().numpy().mean(axis=1)
                scores_dir[pi:pi + len(batch_data)] = loss_recons
            pi = pi + len(batch_data)

        # 排序数组
        sorted_scores = np.sort(scores_dir)
        # 计算前 20% 和后 20% 的索引
        n = len(sorted_scores)
        lower_50_index = int(n * 0.5)  # 前 50% 的索引
        # 获取对应的阈值
        lower_50_threshold = sorted_scores[lower_50_index]
        pose_ids = [int(x) for x in pose_ids]
        with tqdm(total=len(dataset_train),desc="dataset创建中") as pbar:
            for ids,train in enumerate(dataset_train):
                data, mate, scene, label ,path = train
                scene_id = mate[0]
                score = scores_dir[ids]
                if score >lower_50_threshold:
                    label = 0
                    n = [data, mate, scene, label,path]
                    datasets.append(n)
                    nn += 1
                pose_id = int(pose_ids[ids])
                scene_id = int(scene_ids[ids])
                label = -1
                if init == 0:
                    for i in range(len(scene_array)):
                        for j in range(len(array)):
                            relation = search_relation('scene', 'pose', f'scene{i+1}', f'pose{j+1}')
                            # print(f'scene:{scene_arr[i]}\tpose:{i+1}\trelation:{relation}')
                            if relation == 'normal':
                                relation_map[i+1][j+1] = 1
                    init = 1
                for i in range(len(scene_array)):
                    if i + 1 != scene_id:
                        scene = get_scene_feature(
                            f"{args.UNVAD}KG/data/cluster/cluster_scene_centers.txt", f'scene{i + 1}')
                        scene = scene.expand(1, 1, 512)
                        a = [data, mate, scene, label, path]
                        dataset_a.append(a)
                        aa += 1

                pbar.update(1)
    else:
        for start_idx in tqdm(range(0, len(dataset_train), batchsize), desc="制作pose_ids中..."):
            batch_data = dataset_train[start_idx:start_idx + batchsize]
            data, mate, scene, label, path = zip(*batch_data)
            pose_input = torch.stack([torch.tensor(d, dtype=torch.float) for d in data]).to(device)
            with torch.no_grad():
                pose_features = model1(pose_input).reshape(len(batch_data),-1)
            pose_ids_clip = optimized_batch_calculation(pose_features,array)+1
            pose_ids[pi:pi + len(batch_data)] = pose_ids_clip  # 将加1后的值赋值到 pose_ids 对应位置

            path_input = torch.stack([torch.tensor(p.reshape(24, 2), dtype=torch.float) for p in path]).to(device)
            scene_input = torch.stack([s.squeeze(0) for s in scene]).to(device)

            with torch.no_grad():
                data_ori = torch.cat((pose_input.reshape(len(batch_data), -1).to(device), scene_input.reshape(len(batch_data), -1).to(device)), dim=1)
                data_rec = model2(pose_input, path_input, scene_input)
                loss_recons = criterion(data_ori, data_rec).cpu().numpy().mean(axis=1)
                scores_dir[pi:pi + len(batch_data)] = loss_recons
            pi = pi + len(batch_data)

        # 排序数组
        sorted_scores = np.sort(scores_dir)
        # 计算前 20% 和后 20% 的索引
        n = len(sorted_scores)
        lower_50_index = int(n * 0.5)  # 前 50% 的索引
        # 获取对应的阈值
        lower_50_threshold = sorted_scores[lower_50_index]

        pose_ids = [int(x) for x in pose_ids]
        with tqdm(total=len(dataset_train),desc="dataset创建中") as pbar:
            for ids,train in enumerate(dataset_train):
                data, mate, scene, label ,path = train
                scene_id = mate[0]
                score = scores_dir[ids]
                if score >lower_50_threshold:
                    n = [data, mate, scene, label,path]
                    datasets.append(n)
                    nn += 1
                pose_id = pose_ids[ids]
                label = -1
                if args.dataset == 'UBnormal':
                    scene_arr = [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
                    if init == 0:
                        for i in range(len(scene_arr)):
                            for j in range(len(array)):
                                relation = search_relation('scene', 'pose', f'scene{scene_arr[i]}', f'pose{j+1}')
                                # print(f'scene:{scene_arr[i]}\tpose:{i+1}\trelation:{relation}')
                                if relation == 'normal':
                                    relation_map[scene_arr[i]][j+1] = 1
                        init = 1
                    for i in range(len(scene_arr)):
                        if  scene_arr[i] != scene_id:
                            # relation = search_relation('scene', 'pose', f'scene{scene_arr[i]}', f'pose{pose_id}')
                            if relation_map[scene_arr[i]][i+1] == 1:
                                continue
                            scene = torch.load(
                                f"{args.DATA}UBnormal_scene_feature/scene{scene_arr[i]}_features.pth")
                            scene = scene.expand(1, 1, 512)
                            a = [data, mate, scene, label,path]
                            dataset_a.append(a)
                            aa += 1
                elif args.dataset == 'NWPUC':
                    scene_arr = [1, 2, 3, 13, 14, 29, 31, 35, 36, 38, 42, 43, 47, 48, 54, 55, 68, 76, 77, 92, 94, 99, 109,
                                 111, 121, 122, 124, 127, 129,
                                 148, 149, 150, 151, 154, 155, 158, 164, 235, 236, 248, 268, 273, 282]
                    if init == 0:
                        for i in range(len(scene_arr)):
                            for j in range(len(array)):
                                relation = search_relation('scene', 'pose', f'scene{scene_arr[i]}', f'pose{j+1}')
                                # print(f'scene:{scene_arr[i]}\tpose:{i+1}\trelation:{relation}')
                                if relation == 'normal':
                                    relation_map[scene_arr[i]][j+1] = 1
                        init = 1
                    for i in range(len(scene_arr)):
                        if  scene_arr[i] != scene_id:
                            # relation = search_relation('scene', 'pose', f'scene{scene_arr[i]}', f'pose{pose_id}')
                            if relation_map[scene_arr[i]][i+1] == 1:
                                continue
                            scene = torch.load(
                                f"{args.DATA}NWPUC_scene_feature/scene{scene_arr[i]}_features.pth")
                            scene = scene.expand(1, 1, 512)
                            a = [data, mate, scene, label,path]
                            dataset_a.append(a)
                            aa += 1
                elif args.dataset == 'ShanghaiTech':
                    scene_arr = [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13]
                    if init == 0:
                        for i in range(len(scene_arr)):
                            for j in range(len(array)):
                                relation = search_relation('scene', 'pose', f'scene{scene_arr[i]}', f'pose{j+1}')
                                # print(f'scene:{scene_arr[i]}\tpose:{i+1}\trelation:{relation}')
                                if relation == 'normal':
                                    relation_map[scene_arr[i]][j+1] = 1
                        init = 1
                    for i in range(len(scene_arr)):
                        if  scene_arr[i] != scene_id:
                            # relation = search_relation('scene', 'pose', f'scene{scene_arr[i]}', f'pose{pose_id}')
                            if relation_map[scene_arr[i]][i+1] == 1:
                                continue
                            scene = torch.load(
                                f"{args.DATA}ShanghaiTech_scene_feature/scene{scene_arr[i]}_features.pth")
                            scene = scene.expand(1, 1, 512)
                            a = [data, mate, scene, label,path]
                            dataset_a.append(a)
                            aa += 1
                else:
                    print(f"这是什么数据集？：{args.dataset}")

                pbar.update(1)


    random.shuffle(datasets)
    # 使用 CustomDataset 来包装数据列表
    dataset_n_tmp = UbnormalDataset(datasets)
    dataset_a_tmp = UbnormalDataset(dataset_a)
    dataset_t = UbnormalDataset(dataset_t)

    loaders = DataLoader(dataset_n_tmp, **loader_args, shuffle=True, generator=torch.Generator(device='cuda'))
    loader_a = DataLoader(dataset_a_tmp, **loader_args, shuffle=True, generator=torch.Generator(device='cuda'))
    loader_t = DataLoader(dataset_t, **loader_args, shuffle=False, generator=torch.Generator(device='cuda'))

    return datasets, dataset_a,dataset_t, loaders, loader_a, loader_t

def main():
    gen_fusion_dataset_dataloader_2()

if __name__ == '__main__':
    # main()
    scene_arr = [1, 2, 3, 13, 14, 29, 31, 35, 36, 38, 42, 43, 47, 48, 54, 55, 68, 76, 77, 92, 94, 99, 109, 111, 121,
                 122, 124, 127, 129, 148, 149, 150, 151, 154, 155, 158, 164, 235, 236, 248, 268, 273, 282]
    print(len(scene_arr))