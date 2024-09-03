import math

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# torch.set_default_tensor_type('torch.cuda.FloatTensor')


def smooth(arr, lamda1):
    arr2 = torch.zeros_like(arr)
    arr2[:-1] = arr[1:]
    arr2[-1] = arr[-1]
    loss = torch.sum((arr2 - arr) ** 2)
    return lamda1 * loss


def sparsity(arr, lamda2):
    loss = torch.sum(arr)
    return lamda2 * loss

def bceloss(prediction_1, prediction_2, lamda0, gamma=2):
    # 创建标签张量
    label_1 = torch.tensor([0.])  # 第一个数据是normal
    label_2 = torch.tensor([1.])  # 第二个数据是abnormal

    # 转换预测值为tensor，如果它们还不是
    prediction_1 = torch.tensor([prediction_1])
    prediction_2 = torch.tensor([prediction_2])

    # 使用PyTorch的binary_cross_entropy函数计算损失
    loss_1 = F.binary_cross_entropy_with_logits(prediction_1, label_1, reduction='none')
    loss_2 = F.binary_cross_entropy_with_logits(prediction_2, label_2, reduction='none')

    # 加入 Focal Loss 的项
    focal_loss_1 = (1 - torch.sigmoid(prediction_1))**gamma * loss_1
    focal_loss_2 = (torch.sigmoid(prediction_2))**gamma * loss_2

    # 返回加权损失
    return lamda0 * (torch.mean(focal_loss_1) + torch.mean(focal_loss_2))

def ranking(scores, batch_size):
    # loss = torch.tensor(0., requires_grad=True, device=scores.device)
    loss = 0
    scores = scores.squeeze()
    topk_n_values, _ = torch.topk(scores[0:batch_size], k=int(batch_size/8))
    topk_a_values, _ = torch.topk(scores[batch_size:batch_size * 2], k=int(batch_size/8))
    maxn = torch.mean(topk_n_values)
    maxa = torch.mean(topk_a_values)
    rank1 = F.relu(1. - maxa + maxn)
    bce_loss = bceloss(maxn, maxa, 1e-2)
    loss = loss + rank1
    loss = loss + bce_loss
    topk_n_values, _ = torch.topk(scores[0:batch_size], k=4)
    topk_a_values, _ = torch.topk(scores[batch_size:batch_size * 2], k=4)
    maxn = torch.mean(topk_n_values)
    maxa = torch.mean(topk_a_values)
    rank2 = F.relu(1-maxa+maxn*(1e-2))
    loss += rank2
    # loss = loss + smooth(scores[0:batch_size],8e-6) #+smooth(scores[batch_size:batch_size*2],4e-5)
    # loss = loss + smooth(scores[batch_size:batch_size*2],4e-5)
    # loss = loss + sparsity(scores[0:batch_size], 8e-6)
    return loss


import itertools
from tqdm import tqdm

def train(nloader, aloader, model, batch_size, optimizer, device):
    model.train()
    loss_num = 0.

    if len(nloader) > len(aloader):
        aloader = itertools.cycle(aloader)
        num_iterations = len(nloader)
    else:
        nloader = itertools.cycle(nloader)
        num_iterations = len(aloader)

    with torch.set_grad_enabled(True), tqdm(total=num_iterations) as pbar:
        for ninput, ainput in zip(nloader, aloader):
            data_n, _, scene_n, _,path_n = ninput
            data_a, _, scene_a, _,path_a = ainput
            path_n = path_n.reshape(-1,24,2)
            path_a = path_a.reshape(-1,24,2)
            pose = torch.cat((data_n, data_a), 0).to(torch.float).to(device)
            path = torch.cat((path_n, path_a), 0).to(torch.float).to(device)
            scene = torch.cat((scene_n, scene_a), 0).to(torch.float).to(device)
            scene = torch.squeeze(scene, dim=1)
            scores = model(pose,path, scene)
            loss = ranking(scores, batch_size)
            loss_num += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)

    avg_loss = loss_num / num_iterations
    print(f"loss: {avg_loss}")

