import math
import random

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
    label_1 = torch.zeros_like(prediction_1)
    label_2 = torch.ones_like(prediction_2)
    loss_1 = F.binary_cross_entropy_with_logits(prediction_1, label_1, reduction='none')
    loss_2 = F.binary_cross_entropy_with_logits(prediction_2, label_2, reduction='none')
    focal_loss_1 = (1 - torch.sigmoid(prediction_1)) ** gamma * loss_1
    focal_loss_2 = torch.sigmoid(prediction_2) ** gamma * loss_2
    return lamda0 * (torch.mean(focal_loss_1) + torch.mean(focal_loss_2))

def focal_loss(predictions, labels, alpha=0.25, gamma=2.0, lamda=1.0):
    BCE_loss = F.binary_cross_entropy_with_logits(predictions, labels, reduction='none')
    probas = torch.sigmoid(predictions)
    focal_loss = alpha * (1 - probas) ** gamma * BCE_loss
    return lamda * focal_loss.mean()

def ranking(scores, batch_size):
    # loss = torch.tensor(0.0, device=scores.device, requires_grad=True)
    loss = 0
    scores = scores.squeeze()

    topk_n_values, _ = torch.topk(scores[0:batch_size], k=1)
    topk_a_values, _ = torch.topk(scores[batch_size:batch_size * 2], k=1)
    maxn = torch.mean(topk_n_values)
    maxa = torch.mean(topk_a_values)
    rank1 = F.relu(1.0 - maxa + maxn)
    # 1e-2 # 0.740  0.414
    # 0
    bce_loss = bceloss(maxn, maxa, 1e-3)
    loss = loss + rank1 + bce_loss

    topk_n_values, _ = torch.topk(scores[0:batch_size], k=batch_size)
    topk_a_values, _ = torch.topk(scores[batch_size:batch_size * 2], k=int(batch_size / 2))
    maxn = torch.mean(topk_n_values)
    maxa = torch.mean(topk_a_values)
    rank2 = F.relu(1.0 - maxa + maxn)
    loss += rank2 * 1e-1  # 1e-2

    return loss


import itertools
from tqdm import tqdm

def shuffle_and_cycle(loader):
    """打乱数据并循环生成器"""
    data = list(loader)  # 将加载器转换为列表
    random.shuffle(data)  # 打乱顺序
    return itertools.cycle(data)  # 转为无限循环的生成器



def adjust_loaders_with_shuffle(nloader, aloader):
    """动态调整加载器并引入打乱操作"""
    nloader_len = len(nloader)
    aloader_len = len(aloader)

    if nloader_len > aloader_len:
        aloader = shuffle_and_cycle(aloader)  # 对较短的加载器打乱并循环
        num_iterations = nloader_len
    else:
        nloader = shuffle_and_cycle(nloader)  # 对较短的加载器打乱并循环
        num_iterations = aloader_len

    return nloader, aloader, num_iterations

def train(nloader, aloader, model, batch_size, optimizer, device):
    model.train()
    loss_num = 0.
    # nloader, aloader, num_iterations = adjust_loaders_with_shuffle(nloader, aloader)

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
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 防止梯度爆炸
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4f}")
            pbar.update(1)

    avg_loss = loss_num / num_iterations
    print(f"loss: {avg_loss}")
    return avg_loss
