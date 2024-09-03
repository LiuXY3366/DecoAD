from torch.utils.data import DataLoader

import torch.optim as optim
import torch
import os
import random
import numpy as np

from stage1.cluster2kg import txt2array, cal_similarity
from stage2.dataset2 import gen_fusion_dataset_dataloader_2
from stage1.dataset import UbnormalDataset
from stage1.fusion import Model
from stage2.train_UBnormal2 import train, FocalLoss
from stage2.test_UBnormal2 import test
from stage1.knowledge_graph import search_relation
from tqdm import tqdm


def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  #并行gpu


setup_seed(int(2024))  # 1577677170  2023

import stage1.option

def main_UBnormal2(epochs = 0,auc_1 = 0,flag1=0, lr=0.000005, weight_decay=0.000005):
    # setup_seed(int(2024))
    checkpoints = f'/home/liuxinyu/PycharmProjects/KG-VAD/stage{flag1}/ckpt/model'+'{:.5f}.pkl'.format(auc_1)
    max_auc = 0.
    loader_args = {'batch_size': 256, 'num_workers': 0, 'pin_memory': False}
    args = stage1.option.parser.parse_args()
    device = torch.device("cuda:0")  # 将 torch.Tensor 分配到的设备的对象
    # return datasets, dataset_a,dataset_t, loaders, loader_a, loader_t
    datasets, dataset_a,dataset_t, loaders, loader_a, loader_t = gen_fusion_dataset_dataloader_2(auc_1 = auc_1,flag1=flag1)


    # 检查计算机上的可用 CUDA 设备数量
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("No CUDA devices available on this machine.")

    # # 设置要使用的 CUDA 设备编号
    # device_id = 0  # 根据需要更改设备编号
    #
    # # 设置指定的 CUDA 设备
    # torch.cuda.set_device(device_id)

    # 将模型移动到指定的设备
    model = Model().to(device)
    criterion = FocalLoss()
    checkpoint = torch.load(checkpoints)
    model.load_state_dict(checkpoint)
    # 仅加载 getpose 部分的权重
    # getpose_state_dict = {
    #     key.replace('getpose.', ''): value
    #     for key, value in checkpoint.items() if key.startswith('getpose.')
    # }
    # model1 = Model()
    # model1.getpose.load_state_dict(getpose_state_dict)
    # 设置模型为评估模式
    # model1.eval()
    for param in model.parameters():
        param.requires_grad = True

    if not os.path.exists('/home/liuxinyu/PycharmProjects/KG-VAD/stage2/ckpt'):
        os.makedirs('/home/liuxinyu/PycharmProjects/KG-VAD/stage2/ckpt')

    optimizer = optim.Adam(model.parameters(),
                            lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    # auc = test(loader_t, model, device)
    # max_auc = auc
    # print(f'auc:{auc}')
    if epochs == 0:
        epochs = args.max_epoch
    for epoch in range(epochs):
        aa = 0
        nn = 0
        aapp = 0
        dataset_a_tmp = []

        # LXY
        if epoch > 0:
            with tqdm(total=len(dataset_a),desc="dataset迭代中") as pbar:
                for datas in dataset_a:
                    data, mate, scene, label ,path = datas
                    pose_input = torch.from_numpy(data).to(torch.float).to(device)
                    # 刚改
                    path_input = torch.from_numpy(path.reshape(-1, 24, 2)).to(torch.float).to(device)
                    scene_input = torch.from_numpy(np.array(scene)).to(torch.float).to(device)
                    score = model(pose_input.unsqueeze(0), path_input,scene_input)
                    if score > 0.8 and label == 1:
                        ap = [data, mate, scene, label,path]
                        datasets.append(ap)
                        aapp += 1
                    elif score < 0.5 and label == 0:
                        n = [data, mate, scene, label,path]
                        datasets.append(n)
                        nn += 1
                    else:
                        a = [data, mate, scene, label,path]
                        dataset_a_tmp.append(a)
                        aa+=1
                    pbar.update(1)
            datasets_l = UbnormalDataset(datasets)
            dataset_a_tmp_l = UbnormalDataset(dataset_a_tmp)

            loaders = DataLoader(datasets_l, **loader_args, shuffle=True, generator=torch.Generator(device='cuda'))

            dataset_a = dataset_a_tmp
            print(f'aa:{aa}\tnn:{nn}\taapp:{aapp}')

        scheduler.step()
        train(loaders,model, optimizer, device,criterion)
        auc = test(loader_t, model,  device)
        torch.save(model.state_dict(),'/home/liuxinyu/PycharmProjects/KG-VAD/stage2/ckpt/' + 'model' + '{:.5f}.pkl'.format(auc))
        if auc>max_auc:
            # torch.save(model.state_dict(), '/home/liuxinyu/PycharmProjects/KG-VAD/stage2/ckpt/' + 'model' + '{:.5f}.pkl'.format(auc))
            # if epoch % 1 == 0 and not epoch == 0:
            #     torch.save(model.state_dict(), '/home/liuxinyu/PycharmProjects/KG-VAD/stage2/ckpt/' + 'model' + '{:.5f}.pkl'.format(auc))
            #     if os.path.exists('/home/liuxinyu/PycharmProjects/KG-VAD/stage2/ckpt/' + 'model' + '{:.5f}.pkl'.format(max_auc)):
            #         os.remove('/home/liuxinyu/PycharmProjects/KG-VAD/stage2/ckpt/' + 'model' + '{:.5f}.pkl'.format(max_auc))
            max_auc = auc
        print('Epoch {0}/{1}: auc:{2}\tmax_auc:{3}\n'.format(epoch, epochs, auc,max_auc))
        # torch.save(model.state_dict(), './ckpt/' + 'model' + '{:.5f}.pkl'.format(auc))
    if max_auc>auc_1:
        flag = 2
    else:
        max_auc = auc_1
        flag = 1
    return max_auc,flag

if __name__ == '__main__':
    # main_UBnormal2()
    scene_id = 1
    pose_id = 1
    relation = search_relation('scene', 'pose', f'scene{scene_id}', f'pose{pose_id}')
    print(relation)
    if relation != 'normal':
        print('ab')
    elif relation != 'abnormal':
        print('n')
    else:
        print(None)

