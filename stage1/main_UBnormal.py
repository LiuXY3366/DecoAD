from torch.utils.data import DataLoader

import torch.optim as optim
import torch
import os
import random
import numpy as np
from stage1.dataset import gen_fusion_dataset_dataloader
from stage1.fusion import Model
# from stage1.fusion_nopath import Model
from stage1.train_UBnormal import train
from stage1.test_UBnormal import test


def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  #并行gpu


setup_seed(int(42))  # 1577677170  2023

import stage1.option
#
# from utils import Visualizer


# torch.set_default_tensor_type('torch.cuda.FloatTensor')
# torch.set_default_tensor_type()
# viz = Visualizer(env='DeepMIL', use_incoming_socket=False)
# def main_UBnormal(epochs = 50,auc_2=1,flag2=0,lr=0.0005, weight_decay=0.00005):
# def main_UBnormal(epochs = 3000,auc_2=1,flag2=0,lr=0.00001, weight_decay=0.000001):
def main_UBnormal(epochs = 3000,auc_2=0,flag2=0,lr=0.00001, weight_decay=0.000001):
# def main_UBnormal(epochs = 50,auc_2=1,flag2=0,lr=0.01, weight_decay=0.00005):
    max_auc = 0.
    args = stage1.option.parser.parse_args()
    device = torch.device("cuda:0")  # 将 torch.Tensor 分配到的设备的对象
    _, _, _, train_nloader, train_aloader, test_loader = gen_fusion_dataset_dataloader()

    # 检查计算机上的可用 CUDA 设备数量
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("No CUDA devices available on this machine.")

    model = Model().to(device)

    if auc_2 != 0:
        checkpoints = f'/home/liuxinyu/PycharmProjects/KG-VAD/stage{flag2}/ckpt/model'+'{:.5f}.pkl'.format(auc_2)
        # checkpoints = '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/ckpt/model0.72346.pkl'
        checkpoint = torch.load(checkpoints)

        model.load_state_dict(checkpoint)
        for param in model.parameters():
            param.requires_grad = True

    train_nloader = train_nloader
    train_aloader = train_aloader

    if not os.path.exists('/home/liuxinyu/PycharmProjects/KG-VAD/stage1/ckpt'):
        os.makedirs('/home/liuxinyu/PycharmProjects/KG-VAD/stage1/ckpt')

    optimizer = optim.Adam(model.parameters(),
                            lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    # for i in range(10):
    # auc = test(test_loader, model, device)
    # auc = 0
    # print(f'auc:{auc}')
    if epochs == 0:
        epochs = args.max_epoch
    for epoch in range(epochs):
        scheduler.step()
        train(train_nloader, train_aloader, model, args.batch_size, optimizer, device)
        roc,pr = test(test_loader, model,  device)
        torch.save(model.state_dict(),
                   '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/ckpt/' + 'fully_model' + '{:.5f}.pkl'.format(roc)+'___'+'{:.5f}.pkl'.format(pr))
        auc = roc
        if auc>max_auc:
            torch.save(model.state_dict(), '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/ckpt/' + 'model' + '{:.5f}.pkl'.format(auc))
            if epoch % 1 == 0 and not epoch == 0:
                torch.save(model.state_dict(), '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/ckpt/' + 'model' + '{:.5f}.pkl'.format(auc))
                # if os.path.exists('/home/liuxinyu/PycharmProjects/KG-VAD/stage1/ckpt/' + 'model' + '{:.5f}.pkl'.format(max_auc)):
                    # os.remove('/home/liuxinyu/PycharmProjects/KG-VAD/stage1/ckpt/' + 'model' + '{:.5f}.pkl'.format(max_auc))
            max_auc = auc
        print('Epoch {0}/{1}: auc:{2}\tmax_auc:{3}\n'.format(epoch, epochs, auc,max_auc))

    if max_auc>auc_2:
        flag = 1
    else:
        max_auc = auc_2
        flag = 2
    return max_auc,flag


if __name__ == '__main__':
    main_UBnormal()
    # # 创建预训练模型实例
    # pretrained_model = stage1.fusion_nopath.Model()  # 使用您的模型定义
    # pretrained_weights = torch.load('/home/liuxinyu/PycharmProjects/KG-VAD/stage2/ckpt/model0.80244.pkl')
    # pretrained_model.load_state_dict(pretrained_weights)
    #
    # # 创建新模型实例
    # new_model = stage1.fusion.Model()  # 使用相同的模型定义，或者如果有变更，使用新的定义
    # new_weights = torch.load('/home/liuxinyu/PycharmProjects/KG-VAD/stage2/ckpt/model0.79140.pkl')
    # new_model.load_state_dict(new_weights)
    #
    # # 复制兼容的权重（如果需要）
    # pretrained_state_dict = pretrained_model.state_dict()
    # new_state_dict = new_model.state_dict()
    #
    # for name, param in pretrained_state_dict.items():
    #     if name in new_state_dict and new_state_dict[name].size() == param.size():
    #         print(name)
    #         new_state_dict[name].copy_(param)
    #
    # new_model.load_state_dict(new_state_dict)
    #
    # # 冻结特定层
    # layers_to_freeze = ['layer1', 'layer2']  # 替换为实际要冻结的层的名称
    # for name, param in new_model.named_parameters():
    #     if name in layers_to_freeze:
    #         param.requires_grad = False


