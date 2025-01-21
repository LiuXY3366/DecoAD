from datetime import datetime

import torch.optim as optim
import torch
import os
import random
import numpy as np

from UNVAD.stage1.dataset import gen_fusion_dataset_dataloader
from UNVAD.stage1.fusion import Autoencoder
from UNVAD.stage1.train import train
from UNVAD.stage1.test import test
from UNVAD.stage1.args import init_parser

# 初始化解析器
parser = init_parser()

# 解析参数
args = parser.parse_args()


# lr=0.0005, weight_decay=0.00005
def main(epochs = 200,auc_2=0,flag2=0,lr=0.01, weight_decay=0.0005,threshold = 0.00005,train_nloader = None,test_loader = None):
    max_auc = 0.
    max_threshold = 0.
    remain_threshold = threshold
    device = torch.device("cuda:0")  # 将 torch.Tensor 分配到的设备的对象
    if train_nloader == None or test_loader == None:
        _,  _, train_nloader, test_loader = gen_fusion_dataset_dataloader()

    # 检查计算机上的可用 CUDA 设备数量
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("No CUDA devices available on this machine.")

    # 将模型移动到指定的设备
    model = Autoencoder().to(device)

    name = f'{args.supervise}|{args.dataset}'
    if auc_2 != 0:
        checkpoints = f'{args.UNVAD}stage{flag2}/ckpt/{name}/model' + '{:.5f}'.format(auc_2) + '_' + '{:.5f}.pkl'.format(threshold)
        # checkpoints = f'/home/liuxinyu/PycharmProjects/VAD/DecoVAD/UNVAD/stage1/ckpt/U|UBnormal/model0.73191_0.00467.pkl'
        checkpoint = torch.load(checkpoints)

        model.load_state_dict(checkpoint)
        # checkpoint = torch.load(checkpoints)
        # model.load_state_dict(checkpoint,strict=False)
        for param in model.parameters():
            param.requires_grad = True

    train_nloader = train_nloader
    # train_aloader = train_aloader

    if not os.path.exists(f'{args.UNVAD}stage1/ckpt'):
        os.makedirs(f'{args.UNVAD}stage1/ckpt')

    optimizer = optim.Adam(model.parameters(),
                            lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    # 获取当前日期和时间
    current_time = datetime.now().strftime('%Y-%m-%d|%H:%M')

    # 创建文件夹
    folder_name = current_time

    if not os.path.exists(f'{args.UNVAD}stage1/ckpt/data'):
        os.makedirs(f'{args.UNVAD}stage1/ckpt/data')

    os.makedirs(f'{args.UNVAD}stage1/ckpt/data/{args.supervise}|{args.dataset}|{folder_name}', exist_ok=True)


    if not os.path.exists(f'{args.UNVAD}stage1/ckpt/{name}'):
        os.makedirs(f'{args.UNVAD}stage1/ckpt/{name}')
    folder_name = f'{args.supervise}|{args.dataset}|{folder_name}'
    print(f"文件夹已创建: {folder_name}")

    # auc = test(test_loader, model, device,0.5)
    # print(f'auc:{auc}')
    auc = 0.
    if epochs == 0:
        epochs = args.max_epoch
    for epoch in range(epochs):
        optimizer.step()
        scheduler.step()
        _, threshold = train(train_nloader, model, optimizer, device)

        roc,pr = test(test_loader, model, device, threshold)
        torch.save(model.state_dict(),
                   f'{args.UNVAD}stage1/ckpt/data/{folder_name}/' + '{:.5f}'.format(
                       roc) + '_' + '{:.5f}'.format(pr) + '_' + '{:.5f}.pkl'.format(threshold))

        # 注意这里的修改
        auc = roc
        if auc>max_auc:
            torch.save(model.state_dict(), f'{args.UNVAD}stage1/ckpt/{name}/' + 'model' + '{:.5f}'.format(auc) + '_' + '{:.5f}.pkl'.format(threshold))
            if epoch % 1 == 0 and not epoch == 0:
                torch.save(model.state_dict(), f'{args.UNVAD}stage1/ckpt/data/{folder_name}/' + 'model' + '{:.5f}.pkl'.format(auc))
            max_auc = auc
            max_threshold = threshold
        print('Epoch {0}/{1}: auc:{2}\tmax_auc:{3}\n'.format(epoch, epochs, auc,max_auc))

    return_threshold = 0.
    if max_auc>auc_2:
        flag = 1
        return_threshold = max_threshold
    else:
        max_auc = auc_2
        flag = flag2
        return_threshold = remain_threshold
    return max_auc,flag,return_threshold,train_nloader, test_loader




if __name__ == '__main__':
    lr = 0.01
    weight_decay = 0.0005
    for i in range(20):
        lr = lr/(i+1)
        weight_decay = weight_decay /(i+1)
        main(epochs = 5,auc_2=1,flag2=0,lr=lr, weight_decay=weight_decay)
