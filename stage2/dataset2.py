import random
import copy

import numpy as np

from stage1.args import init_parser, init_sub_args
from stage1.cluster2kg import cal_similarity, txt2array
from stage1.dataset import UbnormalDataset,gen_fusion_dataset_dataloader
import torch
from torch.utils.data import DataLoader

from stage1.fusion import Model
from stage1.knowledge_graph import search_relation
from tqdm import tqdm

'''
三个bag：
    纯正bag：在原正bag中存在的pose和scene使用stage1中的模型进行检测，异常分数大于0.8且在知识图谱中的关系是abnormal，重新拼凑的异常分数大于0.9
    正bag:不属于纯正bag和负bag
    负bag:原负bag，以及在原正bag中存在的pose和scene使用stage1中的模型进行检测，异常分数小于0.3且在知识图谱中的关系是normal，重新拼凑的异常分数小于0.1
损失函数：纯正bag和负bag进行二分类，正bag与负bag进行同一阶段相同
'''

# 0.0125
def gen_fusion_dataset_dataloader_2(auc_1 = 0,flag1=0):
    checkpoints = f'/home/liuxinyu/PycharmProjects/KG-VAD/stage{flag1}/ckpt/model'+'{:.5f}.pkl'.format(auc_1)
    device = torch.device("cuda:0")
    model1 = Model()
    model2 = Model()

    # 加载预训练的权重
    checkpoint = torch.load(checkpoints)
    print("Keys in the checkpoint:", checkpoint.keys())

    # 仅加载 getpose 部分的权重
    getpose_state_dict = {
        key.replace('getpose.', ''): value
        for key, value in checkpoint.items() if key.startswith('getpose.')
    }
    model1.getpose.load_state_dict(getpose_state_dict)
    # 设置模型为评估模式
    model1.eval()
    model2.load_state_dict(checkpoint)
    model2.eval()

    datasetn, dataseta, datasett, loader_n, loader_a, loader_t = gen_fusion_dataset_dataloader()
    dataset_train = dataseta
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
    with tqdm(total=len(dataset_train),desc="dataset创建中") as pbar:
        for train in dataset_train:
            data, mate, scene, label ,path = train
            data = data[:2, :, :]
            scene_id = mate[0]
            if label == 1:
                label = 0
                pose_input = torch.from_numpy(data).to(torch.float).to(device)
                path_input = torch.from_numpy(path.reshape(-1, 24, 2)).to(torch.float).to(device)
                scene_input = torch.from_numpy(np.array(scene)).to(torch.float).to(device)
                with torch.no_grad():
                    score = model2(pose_input.unsqueeze(0), path_input, scene_input)
                if score >0.1:
                    n = [data, mate, scene, label,path]
                    datasets.append(n)
                    nn += 1
                continue
            pose_input = torch.from_numpy(data).to(torch.float).to(device)
            path_input = torch.from_numpy(path.reshape(-1, 24, 2)).to(torch.float).to(device)
            scene_input = torch.from_numpy(np.array(scene)).to(torch.float).to(device)
            with torch.no_grad():
                score = model2(pose_input.unsqueeze(0),path_input, scene_input)
                pose_feature = model1(pose_input.unsqueeze(0)).squeeze(0).cpu().detach().numpy()
            array = txt2array()
            pose_id = cal_similarity(pose_feature,array)+1
            # print( f'scene{scene_id}\tpose{pose_id}')
            if old_scene_id == scene_id and old_pose_id == pose_id and (old_relation == 'normal' or old_relation =='abnormal'):
                relation = old_relation
                # print(f'使用old_relation：{relation}')
            else:
                relation = search_relation('scene', 'pose', f'scene{scene_id}', f'pose{pose_id}')
                # print(f'使用new_relation：{relation}')
                old_scene_id = scene_id
                old_pose_id = pose_id
                old_relation = relation
            # print(f'scene{scene_id}\tpose{pose_id}\trelation:{relation}')
            if relation == 'abnormal' and score > 0.8:
                label = 1
                ap = [data, mate, scene, label,path]
                datasets.append(ap)
                aplus+=1
            elif relation == 'normal' and score < 0.5:
                label = 0
                n = [data, mate, scene, label,path]
                datasets.append(n)
                nn += 1
            else:
                if relation == 'normal':
                    a = [data, mate, scene, 0,path]
                    dataset_a.append(a)
                    aa += 1
                elif relation == 'abnormal':
                    a = [data, mate, scene, 1, path]
                    dataset_a.append(a)
                    aa += 1
            # for i in range(13):
            #     if  i+1 != scene_id:
            #         scene = torch.load(
            #             f"/home/liuxinyu/PycharmProjects/KG-VAD/data/NWPUC_scene_feature/scene{i+1}_features.pth")
            #         scene = scene.expand(1, 1, 512)
            #         a = [data, mate, scene, label,path]
            #         dataset_a.append(a)
            #         aa += 1
            scene_arr = [1,2,3,13,14,29,31,35,36,38,42,43,47,48,54,55,68,76,77,92,94,99,109,111,121,122,124,127,129,
                         148,149,150,151,154,155,158,164,235,236,248,268,273,282]
            for i in range(29):
                if  scene_arr[i] != scene_id:
                    scene = torch.load(
                        f"/home/liuxinyu/PycharmProjects/KG-VAD/data/UBnormal_scene_feature/scene{i+1}_features.pth")
                    scene = scene.expand(1, 1, 512)
                    a = [data, mate, scene, label,path]
                    dataset_a.append(a)
                    aa += 1
            # for i in range(len(scene_arr)):
            #     if  scene_arr[i] != scene_id:
            #         scene = torch.load(
            #             f"/home/liuxinyu/PycharmProjects/KG-VAD/data/NWPUC_scene_feature/scene{scene_arr[i]}_features.pth")
            #         scene = scene.expand(1, 1, 512)
            #         a = [data, mate, scene, label,path]
            #         dataset_a.append(a)
            #         aa += 1

            pbar.update(1)

    print(f'nn:{nn}\taa:{aa}\taplus:{aplus}')

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