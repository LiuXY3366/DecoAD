import itertools

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

# # 定义 Focal Loss
# class FocalLoss(torch.nn.Module):
#     def __init__(self, gamma=2, alpha=None, reduction='mean'):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.reduction = reduction
#
#     def forward(self, inputs, targets):
#         ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         pt = torch.exp(-ce_loss)
#         focal_loss = (1 - pt) ** self.gamma * ce_loss
#
#         if self.alpha is not None:
#             focal_loss = self.alpha * focal_loss
#
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss
#
# def smooth(arr, lamda1):
#     arr2 = torch.zeros_like(arr)
#     arr2[:-1] = arr[1:]
#     arr2[-1] = arr[-1]
#     loss = torch.sum((arr2 - arr) ** 2)
#     return lamda1 * loss
#
#
# def sparsity(arr, lamda2):
#     loss = torch.sum(arr)
#     return lamda2 * loss
#
# def bceloss(prediction_1, prediction_2, lamda0, gamma=2):
#     # 创建标签张量
#     label_1 = torch.tensor([0.])  # 第一个数据是normal
#     label_2 = torch.tensor([1.])  # 第二个数据是abnormal
#
#     # 转换预测值为tensor，如果它们还不是
#     prediction_1 = torch.tensor([prediction_1])
#     prediction_2 = torch.tensor([prediction_2])
#
#     # 使用PyTorch的binary_cross_entropy函数计算损失
#     loss_1 = F.binary_cross_entropy_with_logits(prediction_1, label_1, reduction='none')
#     loss_2 = F.binary_cross_entropy_with_logits(prediction_2, label_2, reduction='none')
#
#     # 加入 Focal Loss 的项
#     focal_loss_1 = (1 - torch.sigmoid(prediction_1))**gamma * loss_1
#     focal_loss_2 = (torch.sigmoid(prediction_2))**gamma * loss_2
#
#     # 返回加权损失
#     # return lamda0 * (torch.mean(focal_loss_1) + torch.mean(focal_loss_2))
#     return lamda0 * (torch.mean(focal_loss_1) + torch.mean(focal_loss_2))
#
# def ranking(scores, batch_size):
#     # loss = torch.tensor(0., requires_grad=True, device=scores.device)
#     loss = 0
#     scores = scores.squeeze()
#     # topk_n_values, _ = torch.topk(scores[0:batch_size], k=int(batch_size/8))
#     topk_n_values, _ = torch.topk(scores[0:batch_size], k=1)
#
#     # 执行 topk 操作
#     topk_a_values, _ = torch.topk(scores[batch_size:batch_size * 2], k=1)
#
#     maxn = torch.mean(topk_n_values)
#     maxa = torch.mean(topk_a_values)
#     rank1 = F.relu(1. - maxa + maxn)
#     bce_loss = bceloss(maxn, maxa, 1e-2)
#     loss = loss + rank1
#     loss = loss + bce_loss
#     # topk_n_values, _ = torch.topk(scores[0:batch_size], k=4)
#     # topk_a_values, _ = torch.topk(scores[batch_size:batch_size * 2], k=4)
#     # maxn = torch.mean(topk_n_values)
#     # maxa = torch.mean(topk_a_values)
#     # rank2 = F.relu(1-maxa+maxn*(1e-2))
#     # loss += rank2
#     # loss = loss + smooth(scores[0:batch_size],8e-6) #+smooth(scores[batch_size:batch_size*2],4e-5)
#     # loss = loss + smooth(scores[batch_size:batch_size*2],4e-5)
#     # loss = loss + sparsity(scores[0:batch_size], 8e-6)
#     return loss

# 定义 Focal Loss
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def adjust_loaders1(nloader, aloader):
    nloader_len = len(nloader)
    aloader_len = len(aloader)

    # 确定哪个加载器更长，计算步长
    if nloader_len > aloader_len:
        max_iterations = aloader_len * 2
        if nloader_len <= max_iterations:
            num_iterations = nloader_len
        else:
            step = nloader_len / (aloader_len * 2)
            num_iterations = aloader_len * 2  # 最大迭代次数
            nloader = (nloader[int(i * step)] for i in range(num_iterations))
        aloader = itertools.cycle(aloader)  # 保持生成器特性
    else:
        max_iterations = nloader_len * 2
        if aloader_len <= max_iterations:
            num_iterations = aloader_len
        else:
            step = aloader_len / (nloader_len * 2)
            num_iterations = nloader_len * 2  # 最大迭代次数
            aloader = (aloader[int(i * step)] for i in range(num_iterations))
        nloader = itertools.cycle(nloader)  # 保持生成器特性

    return nloader, aloader, num_iterations

import itertools

def adjust_loaders(nloader, aloader):
    # 获取加载器长度
    nloader_len = len(nloader)
    aloader_len = len(aloader)

    # 确定哪个加载器更长，计算步长
    if nloader_len > aloader_len:
        max_iterations = aloader_len * 2
        if nloader_len <= max_iterations:
            # 如果长加载器不超过短加载器的两倍，就按长加载器的长度来迭代
            num_iterations = nloader_len
            # 对较短的加载器（aloader）使用 itertools.cycle()
            aloader = itertools.cycle(aloader)  # 保持生成器特性
        else:
            # 如果长加载器超过短加载器的两倍，按步长间隔取样
            step = nloader_len / (aloader_len * 2)
            num_iterations = aloader_len * 2  # 最大迭代次数
            # 使用 islice 跳跃访问
            nloader = itertools.islice(nloader, 0, None, int(step))  # 使用 islice 进行间隔取样
            aloader = itertools.cycle(aloader)  # 保持生成器特性
    else:
        max_iterations = nloader_len * 2
        if aloader_len <= max_iterations:
            # 如果长加载器不超过短加载器的两倍，就按长加载器的长度来迭代
            num_iterations = aloader_len
            # 对较短的加载器（nloader）使用 itertools.cycle()
            nloader = itertools.cycle(nloader)  # 保持生成器特性
        else:
            # 如果长加载器超过短加载器的两倍，按步长间隔取样
            step = aloader_len / (nloader_len * 2)
            num_iterations = nloader_len * 2  # 最大迭代次数
            # 使用 islice 跳跃访问
            aloader = itertools.islice(aloader, 0, None, int(step))  # 使用 islice 进行间隔取样
            nloader = itertools.cycle(nloader)  # 保持生成器特性

    return nloader, aloader, num_iterations


def train(loaders, model, batch_size, optimizer, device):
    # 使用 MSELoss 并设置 reduction='none'
    criterion = nn.MSELoss(reduction='none')
    model.train()
    loss_num = 0.
    loss_reco_num = 0.

    with torch.set_grad_enabled(True), tqdm(total=len(loaders)) as pbar:
        for input in loaders:
            pose, _, scene, label, path = input
            # 1. 使用带标签的数据进行监督学习
            pose = pose.to(torch.float).to(device).requires_grad_()
            path = path.reshape(-1, 24, 2).to(torch.float).to(device).requires_grad_()
            scene = scene.to(torch.float).to(device).requires_grad_()
            scene = torch.squeeze(scene, dim=1)

            data_ori = torch.cat((pose.view(pose.size(0), -1), scene.view(pose.size(0), -1)), dim=1)
            data_rec = model(pose, path, scene)
            # print('data_rec.size()')
            # print(data_rec.size())

            # 逐元素损失，输出形状仍为 [512, 1376]
            loss_elementwise = criterion(data_ori, data_rec)
            # 按列求平均，得到形状 [512]
            score = loss_elementwise.mean(dim=1)

            alpha = 0.5
            weights = torch.where(label == 0, torch.tensor(alpha), torch.tensor(1.0 - alpha))
            loss = F.binary_cross_entropy_with_logits(score, label.float(), weight=weights)

            loss_num += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)

    avg_loss = loss_num / len(loaders)
    print(f"loss: {avg_loss}")
    # 确保 all_losses 是张量，且从计算图中分离
    # all_losses = (loss_reco_num / num_iterations).detach().cpu().numpy()

    # 计算 90% 分位数
    # threshold = np.percentile(all_losses, 90)
    return avg_loss


# # 输入张量
# data_ori = torch.randn(512, 1376)  # 形状 [512, 1376]
# data_rec = torch.randn(512, 1376)  # 形状 [512, 1376]
#
# # 使用 MSELoss 并设置 reduction='none'
# criterion = nn.MSELoss(reduction='none')
#
# # 逐元素损失，输出形状仍为 [512, 1376]
# loss_elementwise = criterion(data_ori, data_rec)
# # 按列求平均，得到形状 [512]
# loss_recon = loss_elementwise.mean(dim=1)
# print(loss_recon.shape)  # 输出 [512]