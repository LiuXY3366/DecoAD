import math
from datetime import datetime

from torch import nn
from torch.utils.data import DataLoader

import torch.optim as optim
import torch
import os
import random
import numpy as np

from UNVAD.KG.cluster0 import get_cluster, get_scene_cluster
from UNVAD.KG.cluster2kg import txt2array, cal_similarity, cluster_test, cluster_all_test
from UNVAD.stage2.dataset2 import gen_fusion_dataset_dataloader_2
from UNVAD.stage1.dataset import UbnormalDataset
from UNVAD.stage1.fusion import Autoencoder
from UNVAD.stage2.train2 import train, FocalLoss
from UNVAD.stage1.test import test
from UNVAD.KG.knowledge_graph import search_relation, clean_all, init_anything
from tqdm import tqdm
from UNVAD.stage1.args import init_parser




def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  #并行gpu

criterion = nn.MSELoss(reduction='none')
setup_seed(int(2024))  # 1577677170  2023
# 初始化解析器
parser = init_parser()
# 解析参数
args = parser.parse_args()

def main2(epochs = 0, auc_1 = 0., flag1=0, lr=0.005, weight_decay=0.00005, threshold=0.5,batchsize = args.opbs,dataset=None, dataset_a=None, dataset_t=None, loaders=None, loader_a=None, loader_t=None):
    if lr == 0:
        lr = args.lr2
    if weight_decay == 0:
        weight_decay = args.weight_decay2
    name = f'{args.supervise}|{args.dataset}'
    # test
    checkpoints = f'{args.UNVAD}stage{flag1}/ckpt/{name}/model' + '{:.5f}'.format(auc_1) + '_' + '{:.5f}.pkl'.format(threshold)
    max_auc = 0.
    max_threshold = 0
    remain_threshold = threshold
    loader_args = {'batch_size': 256, 'num_workers': 0, 'pin_memory': False}
    device = torch.device("cuda:0")  # 将 torch.Tensor 分配到的设备的对象
    # return datasets, dataset_a,dataset_t, loaders, loader_a, loader_t
    if dataset == None:
        dataset, dataset_a,dataset_t, loaders, loader_a, loader_t = gen_fusion_dataset_dataloader_2(auc_1 = auc_1,flag1=flag1,threshold1=threshold)


    # 检查计算机上的可用 CUDA 设备数量
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("No CUDA devices available on this machine.")


    # 将模型移动到指定的设备
    model = Autoencoder().to(device)

    checkpoint = torch.load(checkpoints)
    model.load_state_dict(checkpoint)

    for param in model.parameters():
        param.requires_grad = True

    if not os.path.exists(f'{args.UNVAD}stage2/ckpt'):
        os.makedirs(f'{args.UNVAD}stage2/ckpt')

    # 获取当前日期和时间
    current_time = datetime.now().strftime('%Y-%m-%d|%H:%M')

    # 创建文件夹
    folder_name = current_time

    if not os.path.exists(f'{args.UNVAD}stage2/ckpt/data'):
        os.makedirs(f'{args.UNVAD}stage2/ckpt/data')

    os.makedirs(f'{args.UNVAD}stage2/ckpt/data/{args.supervise}|{args.dataset}|{folder_name}', exist_ok=True)


    if not os.path.exists(f'{args.UNVAD}stage2/ckpt/{name}'):
        os.makedirs(f'{args.UNVAD}stage2/ckpt/{name}')
    folder_name = f'{args.supervise}|{args.dataset}|{folder_name}'
    print(f"文件夹已创建: {folder_name}")

    optimizer = optim.Adam(model.parameters(),
                            lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    dataset_abnormal = []
    if epochs == 0:
        epochs = args.max_epoch
    for epoch in range(epochs):
        aa = 0
        nn = 0
        aapp = 0
        dataset_a_tmp = []
        pi = 0
        scores_dir = np.zeros(len(dataset_a))

        if 1 % 5 == 0:
            for start_idx in tqdm(range(0, len(dataset_a), batchsize), desc="制作scores_dir中..."):
                batch_data = dataset_a[start_idx:start_idx + batchsize]
                # data, mate, scene, label, path = batch_data
                data, mate, scene, label, path = zip(*batch_data)
                # pose_input = torch.from_numpy(data).to(torch.float).to(device)
                pose_input = torch.stack([torch.tensor(d, dtype=torch.float) for d in data]).to(device)
                path_input = torch.stack([torch.tensor(p.reshape(24, 2), dtype=torch.float) for p in path]).to(device)
                scene_input = torch.stack([s.squeeze(0) for s in scene]).to(device)
                with torch.no_grad():
                    data_ori = torch.cat((pose_input.reshape(len(batch_data), -1).to(device),
                                          scene_input.reshape(len(batch_data), -1).to(device)), dim=1)
                    data_rec = model(pose_input, path_input, scene_input)
                    loss_recons = criterion(data_ori, data_rec).cpu().numpy().mean(axis=1)
                    scores_dir[pi:pi + len(batch_data)] = loss_recons
                pi = pi + len(batch_data)

            # 排序数组
            sorted_scores = np.sort(scores_dir)

            # 计算前 20% 和后 20% 的索引
            n = len(sorted_scores)
            lower_20_index = int(n * 0.2)  # 前 20% 的索引
            upper_20_index = int(n * 0.8)  # 后 20% 的索引（相对于前 80% 的结束点）

            # 获取对应的阈值
            lower_20_threshold = sorted_scores[lower_20_index]
            upper_20_threshold = sorted_scores[upper_20_index]

            with tqdm(total=len(dataset_a),desc="dataset迭代中") as pbar:
                for i,datas in enumerate(dataset_a):
                    data, mate, scene, label ,path = datas
                    scene = scene.to('cpu')
                    score = scores_dir[i]
                    if score > upper_20_threshold and label == -1:
                        label = 1
                        ap = [data, mate, scene, label,path]
                        dataset.append(ap)
                        aapp += 1
                    elif score < lower_20_threshold:
                        label = 0
                        n = [data, mate, scene, label,path]
                        dataset.append(n)
                        nn += 1
                    else:
                        a = [data, mate, scene, label,path]
                        dataset_a_tmp.append(a)
                        aa+=1
                    pbar.update(1)
            # datasets_normal = UbnormalDataset(dataset_normal)
            datasets_l = UbnormalDataset(dataset)
            # datasets_abnormal = UbnormalDataset(dataset_abnormal)
            print(f'aa:{len(dataset_a_tmp)}\tnn:{nn}\taapp:{aapp}')
            loaders = DataLoader(datasets_l, **loader_args, shuffle=True, generator=torch.Generator(device='cuda'))
            # loaders_abnormal = DataLoader(datasets_abnormal, **loader_args, shuffle=True, generator=torch.Generator(device='cuda'))
            dataset_a = dataset_a_tmp

        optimizer.step()
        scheduler.step()
        train(loaders,model, loader_args['batch_size'],optimizer, device)

        roc, pr = test(loader_t, model, device, threshold)
        torch.save(model.state_dict(),
                   f'{args.UNVAD}stage2/ckpt/data/{folder_name}/' + '{:.5f}'.format(
                       roc) + '_' + '{:.5f}'.format(pr) + '_' + '{:.5f}.pkl'.format(threshold))
        auc = roc
        if auc>max_auc:
            torch.save(model.state_dict(), f'{args.UNVAD}stage2/ckpt/{name}/' + 'model' + '{:.5f}'.format(auc) + '_' + '{:.5f}.pkl'.format(threshold))
            if epoch % 1 == 0 and not epoch == 0:
                torch.save(model.state_dict(), f'{args.UNVAD}stage2/ckpt/data/{folder_name}/' + 'model' + '{:.5f}.pkl'.format(auc))
            max_auc = auc
            max_threshold = threshold
        print('Epoch {0}/{1}: auc:{2}\tmax_auc:{3}\n'.format(epoch, epochs, auc,max_auc))

    return_threshold  = 0.
    if max_auc>auc_1:
        flag = 2
        return_threshold = max_threshold
    else:
        max_auc = auc_1
        flag = flag1
        return_threshold = remain_threshold
    return max_auc,flag,return_threshold

if __name__ == '__main__':
    flag1 = 1
    auc_1 = 0.64186
    threshold1 = 22559.80293
    # pose_num,dataset_input, loader_input = get_cluster(auc_1, flag1,threshold1)
    # clean_all()
    # init_anything(scene=400, pose=pose_num)
    # cluster_test(auc_1, flag1,threshold1)
    #
    # if args.dataset == 'UFSR':
    #     scene_num = get_scene_cluster(40)
    #     clean_all()
    #     init_anything(scene=40, pose=pose_num)
    #     cluster_all_test(auc_1, flag1,threshold1, dataset_input, loader_input)
    # else:
    #     clean_all()
    #     init_anything(scene=400, pose=pose_num)
    #     cluster_test(auc_1, flag1,threshold1, dataset_input, loader_input)
    # print('以上运行结束，没有报错！')
    main2(auc_1=auc_1,flag1=flag1,threshold = threshold1 )

    # main2()

