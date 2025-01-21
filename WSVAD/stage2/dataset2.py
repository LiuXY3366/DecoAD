import random
import copy

import numpy as np

from WSVAD.KG.cluster2kg import cal_similarity, txt2array, optimized_batch_calculation
from WSVAD.stage1.dataset import UbnormalDataset, gen_fusion_dataset_dataloader, get_scene_feature
import torch
from torch.utils.data import DataLoader

from WSVAD.stage1.fusion import Model
from WSVAD.KG.knowledge_graph import search_relation
from tqdm import tqdm
from WSVAD.stage1.args import init_parser

# 初始化解析器
parser = init_parser()

# 解析参数
args = parser.parse_args()

'''
三个bag：
    纯正bag：在原正bag中存在的pose和scene使用stage1中的模型进行检测，异常分数大于0.8且在知识图谱中的关系是abnormal，重新拼凑的异常分数大于0.9
    正bag:不属于纯正bag和负bag
    负bag:原负bag，以及在原正bag中存在的pose和scene使用stage1中的模型进行检测，异常分数小于0.3且在知识图谱中的关系是normal，重新拼凑的异常分数小于0.1
损失函数：纯正bag和负bag进行二分类，正bag与负bag进行同一阶段相同
'''

# 0.0125
def gen_fusion_dataset_dataloader_2(auc_1 = 0,flag1=0,batchsize = args.opbs):
    name = f'{args.supervise}|{args.dataset}'
    checkpoints = f'{args.WSVAD}stage{flag1}/ckpt/{name}/model' + '{:.5f}.pkl'.format(
        auc_1)
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
    array = txt2array()
    pose_ids = np.zeros(len(dataset_train))
    scene_ids = np.zeros(len(dataset_train))
    pi = 0
    scores_dir = np.zeros(len(dataset_train))
    init = 0
    relation_map = np.zeros((1000, 1000))

    if args.dataset == 'UFSR':
        scene_array = txt2array('scene')
        for start_idx in tqdm(range(0, len(dataset_train), batchsize), desc="制作pose_ids中..."):
            batch_data = dataset_train[start_idx:start_idx + batchsize]
            # data, mate, scene, label, path = batch_data
            data, mate, scene, label, path = zip(*batch_data)
            # print(f'scene:{scene}')
            # pose_input = torch.from_numpy(data).to(torch.float).to(device)
            pose_input = torch.stack([torch.tensor(d, dtype=torch.float) for d in data]).to(device)
            scene_features = torch.stack([torch.tensor(s, dtype=torch.float) for s in scene]).reshape(len(batch_data), -1).to(device)
            with torch.no_grad():
                pose_features = model1(pose_input).reshape(len(batch_data), -1)
            pose_ids_clip = optimized_batch_calculation(pose_features, array) + 1
            scene_ids_clip = optimized_batch_calculation(scene_features, scene_array) + 1
            pose_ids[pi:pi + len(batch_data)] = pose_ids_clip  # 将加1后的值赋值到 pose_ids 对应位置
            scene_ids[pi:pi + len(batch_data)] = scene_ids_clip  # 将加1后的值赋值到 pose_ids 对应位置

            path_input = torch.stack([torch.tensor(p.reshape(24, 2), dtype=torch.float) for p in path]).to(device)
            # scene_input = torch.stack([torch.tensor(s.squeeze(0), dtype=torch.float) for s in scene]).to(device)
            scene_input = torch.stack([s.squeeze(0) for s in scene]).to(device)
            with torch.no_grad():
                scores = model2(pose_input, path_input, scene_input)
                scores_dir[pi:pi + len(batch_data)] = scores.cpu().squeeze(1)
            pi = pi + len(batch_data)

        # 排序数组
        sorted_scores = np.sort(scores_dir)
        # 计算前 20% 和后 20% 的索引
        n = len(sorted_scores)
        lower_10_index = int(n * 0.1)
        lower_5_index = int(n * 0.5)
        lower_95_index = int(n * 0.95)
        lower_40_index = int(n * 0.4)
        lower_60_index = int(n * 0.6)


        with tqdm(total=len(dataset_train), desc="dataset创建中") as pbar:
            for ids, train in enumerate(dataset_train):
                # print(f'nn:{nn}\taa:{aa}\taplus:{aplus}')
                data, mate, scene, label, path = train
                data = data[:2, :, :]
                # scene_id = mate[0]
                score = scores_dir[ids]
                if label == 1:
                    label = 0
                    if score > sorted_scores[lower_10_index]:
                        n = [data, mate, scene, label, path]
                        datasets.append(n)
                        nn += 1
                    continue
                pose_id = int(pose_ids[ids])
                scene_id = int(scene_ids[ids])
                if old_scene_id == scene_id and old_pose_id == pose_id and (
                        old_relation == 'normal' or old_relation == 'abnormal'):
                    relation = old_relation
                else:
                    relation = search_relation('scene', 'pose', f'scene{scene_id}', f'pose{pose_id}')
                    old_scene_id = scene_id
                    old_pose_id = pose_id
                    old_relation = relation

                # if relation == 'abnormal' and score > sorted_scores[lower_80_index]:
                if score > sorted_scores[lower_95_index] or (relation == 'abnormal' and score > sorted_scores[lower_60_index]):
                    label = 1
                    ap = [data, mate, scene, label, path]
                    datasets.append(ap)
                    aplus += 1
                # elif relation == 'normal' and score < sorted_scores[lower_20_index]:
                elif score < sorted_scores[lower_5_index] or (relation == 'normal' and score < sorted_scores[lower_40_index]):
                    label = 0
                    n = [data, mate, scene, label, path]
                    datasets.append(n)
                    nn += 1
                else:
                    if relation == 'normal':
                        a = [data, mate, scene, 0, path]
                        dataset_a.append(a)
                        aa += 1
                    elif relation == 'abnormal':
                        a = [data, mate, scene, 1, path]
                        dataset_a.append(a)
                        aa += 1

                for i in range(len(scene_array)):
                    if i + 1 != scene_id:
                        scene = get_scene_feature(
                            f"{args.WSVAD}KG/data/cluster/cluster_scene_centers.txt", f'scene{i + 1}')
                        # scene = torch.load(
                        #     f"{args.DATA}UFSR_scene_feature/scene{i + 1}_features.pth")
                        scene = scene.expand(1, 1, 512)
                        a = [data, mate, scene, label, path]
                        dataset_a.append(a)
                        aa += 1

                pbar.update(1)
    else:
        for start_idx in tqdm(range(0, len(dataset_train), batchsize), desc="制作pose_ids中..."):
            batch_data = dataset_train[start_idx:start_idx + batchsize]
            # data, mate, scene, label, path = batch_data
            data, mate, scene, label, path = zip(*batch_data)
            # pose_input = torch.from_numpy(data).to(torch.float).to(device)
            pose_input = torch.stack([torch.tensor(d, dtype=torch.float) for d in data]).to(device)
            with torch.no_grad():
                pose_features = model1(pose_input).reshape(len(batch_data),-1)
            pose_ids_clip = optimized_batch_calculation(pose_features,array)+1
            pose_ids[pi:pi + len(batch_data)] = pose_ids_clip  # 将加1后的值赋值到 pose_ids 对应位置

            path_input = torch.stack([torch.tensor(p.reshape(24, 2), dtype=torch.float) for p in path]).to(device)
            # scene_input = torch.stack([torch.tensor(s.squeeze(0), dtype=torch.float) for s in scene]).to(device)
            scene_input = torch.stack([s.squeeze(0) for s in scene]).to(device)
            with torch.no_grad():
                scores = model2(pose_input, path_input, scene_input)
                scores_dir[pi:pi + len(batch_data)] = scores.cpu().squeeze(1)
            pi = pi + len(batch_data)

        # 排序数组
        sorted_scores = np.sort(scores_dir)
        # 计算前 20% 和后 20% 的索引
        n = len(sorted_scores)
        lower_10_index = int(n * 0.1)
        lower_20_index = int(n * 0.2)
        lower_80_index = int(n * 0.8)

        with tqdm(total=len(dataset_train),desc="dataset创建中") as pbar:
            for ids, train in enumerate(dataset_train):

                data, mate, scene, label ,path = train
                data = data[:2, :, :]
                scene_id = mate[0]
                score = scores_dir[ids]
                if label == 1:
                    label = 0
                    if score >sorted_scores[lower_10_index]:
                        n = [data, mate, scene, label,path]
                        datasets.append(n)
                        nn += 1
                    continue
                pose_id = int(pose_ids[ids])
                if old_scene_id == scene_id and old_pose_id == pose_id and (old_relation == 'normal' or old_relation =='abnormal'):
                    relation = old_relation
                else:
                    relation = search_relation('scene', 'pose', f'scene{scene_id}', f'pose{pose_id}')
                    old_scene_id = scene_id
                    old_pose_id = pose_id
                    old_relation = relation
                if relation == 'abnormal' and score > sorted_scores[lower_20_index]:
                    label = 1
                    ap = [data, mate, scene, label,path]
                    datasets.append(ap)
                    aplus+=1
                elif relation == 'normal' and score < sorted_scores[lower_80_index]:
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

                if args.dataset == 'UBnormal':
                    for i in range(29):
                        if  i+1 != scene_id:
                            scene = torch.load(
                                f"{args.DATA}UBnormal_scene_feature/scene{i+1}_features.pth")
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
                                elif relation == 'abnormal':
                                    relation_map[scene_arr[i]][j+1] = -1
                        init = 1
                    for i in range(len(scene_arr)):
                        # print(f"scene_arr[i]:{scene_arr[i]}")
                        # print(f"pose_id:{pose_id}")
                        if relation_map[scene_arr[i]][pose_id] == 0:
                        # if  scene_arr[i] != scene_id:
                            scene = torch.load(
                                f"{args.DATA}NWPUC_scene_feature/scene{scene_arr[i]}_features.pth")
                            scene = scene.expand(1, 1, 512)
                            a = [data, mate, scene, label,path]
                            dataset_a.append(a)
                            aa += 1
                else:
                    print(f"这是什么数据集？：{args.dataset}")

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