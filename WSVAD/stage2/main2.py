from datetime import datetime

from torch.utils.data import DataLoader

import torch.optim as optim
import torch
import os
import random
import numpy as np

from WSVAD.KG.cluster0 import get_cluster, get_scene_cluster
from WSVAD.KG.cluster2kg import cluster_test, cluster_all_test
from WSVAD.stage2.dataset2 import gen_fusion_dataset_dataloader_2
from WSVAD.stage1.dataset import UbnormalDataset
from WSVAD.stage1.fusion import Model
from WSVAD.stage2.train2 import train, FocalLoss
from WSVAD.stage1.test import test
from WSVAD.KG.knowledge_graph import clean_all, init_anything
from tqdm import tqdm
from WSVAD.stage1.args import init_parser




def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  #并行gpu


setup_seed(int(2024))  # 1577677170  2023

# 初始化解析器
parser = init_parser()
# 解析参数
args = parser.parse_args()

def main2(epochs = 0,auc_1 = 0.,flag1=0, lr=0.000005, weight_decay=0.000005,datasets=None, dataset_a=None, dataset_t=None, loaders=None, loader_a=None, loader_t=None,batchsize = args.opbs):

    if lr == 0:
        lr = args.lr2
    if weight_decay == 0:
        weight_decay = args.weight_decay2
    name = f'{args.supervise}|{args.dataset}'
    checkpoints = f'{args.WSVAD}/stage{flag1}/ckpt/{name}/model' + '{:.5f}.pkl'.format(
        auc_1)
    max_auc = 0.
    loader_args = {'batch_size': 256, 'num_workers': 0, 'pin_memory': False}
    device = torch.device("cuda:0")  # 将 torch.Tensor 分配到的设备的对象
    # return datasets, dataset_a,dataset_t, loaders, loader_a, loader_t
    if datasets==None:
        datasets, dataset_a,dataset_t, loaders, loader_a, loader_t = gen_fusion_dataset_dataloader_2(auc_1 = auc_1,flag1=flag1)


    # 检查计算机上的可用 CUDA 设备数量
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("No CUDA devices available on this machine.")


    # 将模型移动到指定的设备
    model = Model().to(device)
    criterion = FocalLoss()
    checkpoint = torch.load(checkpoints)
    model.load_state_dict(checkpoint)

    for param in model.parameters():
        param.requires_grad = True

    if not os.path.exists(f'{args.WSVAD}stage2/ckpt'):
        os.makedirs(f'{args.WSVAD}stage2/ckpt')

    # 获取当前日期和时间
    current_time = datetime.now().strftime('%Y-%m-%d|%H:%M')

    # 创建文件夹
    folder_name = current_time

    if not os.path.exists(f'{args.WSVAD}stage2/ckpt/data'):
        os.makedirs(f'{args.WSVAD}stage2/ckpt/data')

    os.makedirs(f'{args.WSVAD}stage2/ckpt/data/{args.supervise}|{args.dataset}|{folder_name}', exist_ok=True)


    if not os.path.exists(f'{args.WSVAD}stage2/ckpt/{name}'):
        os.makedirs(f'{args.WSVAD}stage2/ckpt/{name}')
    folder_name = f'{args.supervise}|{args.dataset}|{folder_name}'
    print(f"文件夹已创建: {folder_name}")

    optimizer = optim.Adam(model.parameters(),
                            lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    if epochs == 0:
        epochs = args.max_epoch
    for epoch in range(epochs):
        aa = 0
        nn = 0
        aapp = 0
        pi = 0
        dataset_a_tmp = []
        scores_dir = np.zeros(len(dataset_a))

        # if epoch > 0:
        if epoch % 5 == 0:
            for start_idx in tqdm(range(0, len(dataset_a), batchsize), desc="制作scores_dir中..."):
                batch_data = dataset_a[start_idx:start_idx + batchsize]
                # data, mate, scene, label, path = batch_data
                data, mate, scene, label, path = zip(*batch_data)

                data = [torch.tensor(d, dtype=torch.float).to('cpu') for d in data]
                pose_input = torch.stack([d for d in data]).to(device)
                path = [torch.tensor(p.reshape(24, 2), dtype=torch.float).to('cpu') for p in path]
                path_input = torch.stack([p for p in path]).to(device)
                scene = [torch.tensor(s, dtype=torch.float).to('cpu') for s in scene]
                scene_input = torch.stack([s.squeeze(0) for s in scene]).to(device)
                with torch.no_grad():
                    logits = model(pose_input, path_input, scene_input)
                    scores_dir[pi:pi + len(batch_data)] = logits.cpu().squeeze(1)
                pi = pi + len(batch_data)

            # 排序数组
            sorted_scores = np.sort(scores_dir)
            # 计算前 20% 和后 20% 的索引
            n = len(sorted_scores)
            lower_10_index = int(n * 0.1)
            lower_5_index = int(n * 0.02)
            lower_95_index = int(n * 0.98)
            lower_40_index = int(n * 0.1)
            lower_60_index = int(n * 0.9)
            with tqdm(total=len(dataset_a),desc="dataset迭代中") as pbar:
                for i, datas in enumerate(dataset_a):
                    data, mate, scene, label ,path = datas
                    scene = scene.to('cpu')
                    score = scores_dir[i]
                    # if score > sorted_scores[lower_90_index]:
                    if (label != 0 and score > sorted_scores[lower_95_index]) or (label == 1 and score > sorted_scores[lower_60_index]):
                        label = 1
                        ap = [data, mate, scene, label,path]
                        datasets.append(ap)
                        aapp += 1
                    # elif score < sorted_scores[lower_10_index]:
                    elif (label != 1 and score < sorted_scores[lower_5_index]) or (label == 0 and score < sorted_scores[lower_40_index]):
                        label = 0
                        n = [data, mate, scene, label,path]
                        datasets.append(n)
                        nn += 1
                    else:
                        a = [data, mate, scene, label,path]
                        dataset_a_tmp.append(a)
                        aa+=1
                    pbar.update(1)

            # with tqdm(total=len(datasets),desc="dataset检查") as pbar:
            #     for i, datas in enumerate(datasets):
            #         data, mate, scene, label, path = datas
            #         print(f"data type: {type(data)}")
            #         print(f"mate type: {type(mate)}")
            #         print(f"scene type: {type(scene)}")
            #         print(f"label type: {type(label)}")
            #         print(f"path type: {type(path)}")

            datasets_l = UbnormalDataset(datasets)
            dataset_a_tmp_l = UbnormalDataset(dataset_a_tmp)

            loaders = DataLoader(datasets_l, **loader_args, shuffle=True, generator=torch.Generator(device='cuda'))

            dataset_a = dataset_a_tmp
            print(f'aa:{aa}\tnn:{nn}\taapp:{aapp}')

        optimizer.step()
        scheduler.step()
        train(loaders,model, optimizer, device,criterion)
        roc,pr = test(loader_t, model,  device)

        torch.save(model.state_dict(),
                   f'{args.WSVAD}stage2/ckpt/data/{folder_name}/'+'{:.5f}'.format(roc)+'_'+'{:.5f}.pkl'.format(pr))

        auc = roc
        if auc>max_auc:
            torch.save(model.state_dict(), f'{args.WSVAD}stage2/ckpt/{name}/' + 'model' + '{:.5f}.pkl'.format(auc))
            if epoch % 1 == 0 and not epoch == 0:
                torch.save(model.state_dict(), f'{args.WSVAD}stage2/ckpt/data/{folder_name}/' + 'model' + '{:.5f}.pkl'.format(auc))
            max_auc = auc
        print('Epoch {0}/{1}: auc:{2}\tmax_auc:{3}\n'.format(epoch, epochs, auc,max_auc))

    if max_auc>auc_1:
        flag = 2
    else:
        max_auc = auc_1
        flag = flag1
    return round(max_auc, 5),flag

if __name__ == '__main__':
    auc_1 = 0.75930
    flag1 = 2
    pose_num, dataset_input, loader_input = get_cluster(auc_1, flag1)
    if args.dataset == 'UFSR':
        scene_num = get_scene_cluster(40)
        clean_all()
        init_anything(scene=40, pose=pose_num)
        cluster_all_test(auc_1, flag1, dataset_input, loader_input)
    else:
        clean_all()
        init_anything(scene=400, pose=pose_num)
        cluster_test(auc_1, flag1, dataset_input, loader_input)
    print('以上运行结束，没有报错！')
    main2(auc_1=auc_1,flag1=flag1)

