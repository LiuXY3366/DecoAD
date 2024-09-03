import torch
import torch.nn.functional as F
from tqdm import tqdm

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

def train(loaders, model, optimizer, device,criterion):
    model.train()
    loss_num = 0.

    with torch.set_grad_enabled(True), tqdm(total=len(loaders)) as pbar:
        for input in loaders:
            pose, _, scene, label,path = input

            # 1. 使用带标签的数据进行监督学习
            pose = pose.to(torch.float).to(device).requires_grad_()
            path = path.reshape(-1,24,2).to(torch.float).to(device).requires_grad_()
            scene = scene.to(torch.float).to(device).requires_grad_()
            scene = torch.squeeze(scene, dim=1)

            score = model(pose,path,scene).squeeze(1)

            alpha = 1e-1
            weights = torch.where(label == 0, torch.tensor(alpha), torch.tensor(1.0-alpha))
            loss = F.binary_cross_entropy_with_logits(score, label.float(), weight=weights)
            loss_num += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(1)

    avg_loss = loss_num / len(loaders)
    print(f"loss: {avg_loss}")
