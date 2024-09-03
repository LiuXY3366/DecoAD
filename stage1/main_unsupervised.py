import torch.optim as optim
import torch
import os
import random
import numpy as np

from stage1.dataset import gen_fusion_dataset_dataloader
from stage1.fusion_unsuperivesd import Autoencoder
from stage1.train_unsupervised import train
from stage1.test_unsupervised import test


# def setup_seed(seed):
#     random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)  # cpu
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  #并行gpu
#
#
# setup_seed(int(42))  # 1577677170  2023

import stage1.option
#
# from utils import Visualizer


# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# torch.set_default_tensor_type()
# viz = Visualizer(env='DeepMIL', use_incoming_socket=False)
# def main_UBnormal(epochs = 300,auc_2=0,flag2=0,lr=0.0005, weight_decay=0.00005):
def main_STC(epochs = 300,auc_2=1,flag2=0,lr=0.001, weight_decay=0.00005):
# def main_NWPUC(epochs = 300,auc_2=0,flag2=0,lr=0.0001, weight_decay=0.00005):
# def main_NWPUC(epochs = 3000,auc_2=1,flag2=0,lr=0.000001, weight_decay=0):
# def main_NWPUC(epochs = 3000,auc_2=1,flag2=0,lr=0.5, weight_decay=0.005):  NWPUC
    max_auc = 0.
    args = stage1.option.parser.parse_args()
    device = torch.device("cuda:0")  # 将 torch.Tensor 分配到的设备的对象
    _, _, _, train_nloader, train_aloader, test_loader = gen_fusion_dataset_dataloader()

    # 检查计算机上的可用 CUDA 设备数量
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("No CUDA devices available on this machine.")

    # 将模型移动到指定的设备
    model = Autoencoder().to(device)

    if auc_2 != 0:
        checkpoints = f'/home/liuxinyu/PycharmProjects/KG-VAD-STC/stage{flag2}/ckpt/model'+'{:.5f}.pkl'.format(auc_2)
        checkpoints = '/home/liuxinyu/PycharmProjects/KG-VAD-STC/stage1/ckpt/model0.81327.pkl'
        checkpoints = '/home/liuxinyu/PycharmProjects/KG-VAD-STC/stage_UB/stage1/ckpt/modelthreshold0.005893138023093345_auc0.72785.pkl'
        checkpoint = torch.load(checkpoints)

        model.load_state_dict(checkpoint)
        for param in model.parameters():
            param.requires_grad = True

    train_nloader = train_nloader
    # train_aloader = train_aloader

    if not os.path.exists('/home/liuxinyu/PycharmProjects/KG-VAD-STC/stage1/ckpt'):
        os.makedirs('/home/liuxinyu/PycharmProjects/KG-VAD-STC/stage1/ckpt')

    optimizer = optim.Adam(model.parameters(),
                            lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)
    auc = test(test_loader, model, device,0.5)
    print(f'auc:{auc}')
    auc = 0.
    if epochs == 0:
        epochs = args.max_epoch
    fff = 0.67766
    for epoch in range(epochs):
        scheduler.step()
        # if max_auc > fff:
        #     fff = max_auc
        #     print('成功了！！！')
        _, threshold = train(train_nloader, model, optimizer, device)
        auc = test(test_loader, model, device, threshold,dataset_name = 'ShanghaiTech')
        # else:
        #     for k in range(1,10):
        #         _,threshold = train(train_nloader,model, optimizer, device)
        #     auc = test(test_loader, model, device, threshold,dataset_name = 'ShanghaiTech')
        # def test(dataloader, model, device, sl=16, dataset_name='ShanghaiTech'):
        # auc = 0
        # auc = test(test_loader, model, device, threshold)
        # for i in range (1,25):
        #     print(f'================={i}==================')
        #     auc = test(test_loader, model,  device,threshold,i)
        if auc>max_auc:
            torch.save(model.state_dict(), '/home/liuxinyu/PycharmProjects/KG-VAD-STC/stage1/ckpt/' + 'model' + '{:.5f}.pkl'.format(auc))
            if epoch % 1 == 0 and not epoch == 0:
                torch.save(model.state_dict(), '/home/liuxinyu/PycharmProjects/KG-VAD-STC/stage1/ckpt/' + 'model' + '{:.5f}.pkl'.format(auc))
                if os.path.exists('/home/liuxinyu/PycharmProjects/KG-VAD-STC/stage1/ckpt/' + 'model' + '{:.5f}.pkl'.format(max_auc)):
                    os.remove('/home/liuxinyu/PycharmProjects/KG-VAD-STC/stage1/ckpt/' + 'model' + '{:.5f}.pkl'.format(max_auc))
            max_auc = auc
        print('Epoch {0}/{1}: auc:{2}\tmax_auc:{3}\n'.format(epoch, epochs, auc,max_auc))

    if max_auc>auc_2:
        flag = 1
    else:
        max_auc = auc_2
        flag = 2
    return max_auc,flag




if __name__ == '__main__':
    main_STC()
