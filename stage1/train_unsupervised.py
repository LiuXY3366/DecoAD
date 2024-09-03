import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


# def train(nloader, model, optimizer, device):
#     criterion = nn.MSELoss()
#     model.train()
#     all_losses = []
#     loss_num = 0.
#
#     with torch.set_grad_enabled(True), tqdm(total=len(nloader)) as pbar:
#         for ninput in nloader:
#             data_n, _, scene_n, _ = ninput
#             pose = data_n.to(torch.float).to(device)
#             scene = scene_n.to(torch.float).to(device)
#             scene = torch.squeeze(scene, dim=1)
#
#             # 使用动态批处理大小
#             batch_size = pose.size(0)
#             data_ori = torch.cat((pose.view(batch_size, -1), scene.view(batch_size, -1)), dim=1)
#
#             data_rec = model(pose, scene)
#
#             loss = criterion(data_ori, data_rec)
#             all_losses.append(loss.item())
#             loss_num += loss.item()
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             pbar.update(1)
#
#     avg_loss = loss_num / len(nloader)
#     print(f"平均损失: {avg_loss}")
#     # threshold = np.percentile(all_losses, 99)  # 例如，选择99%分位数作为阈值
#     threshold = np.percentile(all_losses, 10)  # 例如，选择90%分位数作为阈值
#     return  avg_loss,threshold

def train(nloader, model, optimizer, device):
    criterion = nn.MSELoss()
    model.train()
    all_losses = []
    loss_num = 0.
    lambda_action = 0.1  # 动作一致性损失的权重
    lambda_sparse = 0.01  # 稀疏损失的权重

    with torch.set_grad_enabled(True), tqdm(total=len(nloader)) as pbar:
        for ninput in nloader:
            data_n, _, scene_n, _ = ninput
            pose = data_n.to(torch.float).to(device)
            scene = scene_n.to(torch.float).to(device)
            scene = torch.squeeze(scene, dim=1)

            batch_size = pose.size(0)
            data_ori = torch.cat((pose.view(batch_size, -1), scene.view(batch_size, -1)), dim=1)

            data_rec = model(pose, scene)

            # 重构损失
            loss_recon = criterion(data_ori, data_rec)

            # 动作一致性损失
            loss_action = torch.mean((data_ori[:, 1:] - data_ori[:, :-1]) - (data_rec[:, 1:] - data_rec[:, :-1])) ** 2

            # 稀疏损失 - 假设 model.encoder 是编码器部分且其输出是编码层的激活值
            encoded_features = model.encoder(pose, scene)
            loss_sparse = torch.mean(torch.abs(encoded_features))

            # 总损失
            loss = loss_recon + lambda_action * loss_action + lambda_sparse * loss_sparse

            all_losses.append(loss.item())
            loss_num += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)

        avg_loss = loss_num / len(nloader)
        print(f"平均损失: {avg_loss}")
        # 选择一个适当的阈值来判断异常值，这里以平均损失为例
        threshold = np.percentile(all_losses, 90)  # 选择90%分位数作为阈值
        return avg_loss, threshold





