import math

import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
import numpy as np
import os

from torch import nn
from tqdm import tqdm

from UNVAD.stage1.dataset import gen_fusion_dataset_dataloader
from UNVAD.stage1.fusion import Autoencoder

from UNVAD.stage1.args import init_parser

parser = init_parser()

# 解析参数
args = parser.parse_args()


def split_str(str):
    first_part,second_part = str.split("_",1)
    return first_part,second_part

def get_gt(gt_path):
    if args.dataset == 'UBnormal':
        # 打开二进制文件并读取数据
        with open(gt_path, 'rb') as file:
            # 读取第一行并丢弃
            file.readline()

            # 读取二进制数据
            numpy_data = np.fromfile(file, dtype=np.float64)
            # print(numpy_data)
            numpy_data = 1 - numpy_data
    else:
        numpy_data = np.load(gt_path)

    return numpy_data.shape[0],numpy_data

def map_loss_to_score(loss_arr, max_loss, threshold, thr=0.5):
    # 创建原数组的副本
    mapped_loss_arr = loss_arr.copy()

    for i in range(len(mapped_loss_arr)):
        loss = mapped_loss_arr[i]
        if loss <= threshold:
            # 映射到0到0.5之间
            mapped_loss_arr[i] = thr * (loss / threshold)
        else:
            # 映射到0.5到1之间
            mapped_loss_arr[i] = thr + thr * ((loss - threshold) / (max_loss - threshold))

    # 返回修改后的数组副本，原始数组不变
    return mapped_loss_arr
from scipy.ndimage import gaussian_filter1d

def gaussian_smooth(arr, sigma=1):
    """高斯滤波器"""
    return gaussian_filter1d(arr, sigma=sigma)


def smooth_scores(scores, window_size=12):
    # 检查窗口大小
    if window_size < 1:
        raise ValueError("Window size must be greater than or equal to 1.")

    # 使用滑动窗口平滑分数
    smoothed_scores = np.convolve(scores, np.ones(window_size) / window_size, mode='same')

    return smoothed_scores

def gaussian_filter_(support, sigma):
    mu = support[len(support) // 2 - 1]
    filter = 1.0 / (sigma * np.sqrt(2 * math.pi)) * np.exp(-0.5 * ((support - mu) / sigma) ** 2)
    return filter 


def test(dataloader, model, device,threshold,sl=24,window_size = 12, smoothing = True):
    true_num = 0
    n_num = 0
    gt_dir = f'{args.data_dir}{args.dataset}/gt/'
    old_mate_data = ''
    old_scene = 0
    num = 0
    gt_zip = []
    pre_zip = []
    max_loss = 0.
    with torch.no_grad():
        criterion = nn.MSELoss(reduction='none')
        model.eval()

        with tqdm(total=len(dataloader)) as pbar:
            for pose_np,mate,scene_np,_,path_data  in dataloader:
                # pose_input = torch.stack([torch.tensor(d.clone().detach(), dtype=torch.float) for d in pose_np]).to(
                #     device)
                pose_input = torch.stack([d.clone().detach().to(torch.float) for d in pose_np]).to(device)

                # path_input = torch.stack(
                #     [torch.tensor(p.reshape(24, 2).clone().detach(), dtype=torch.float) for p in path_data]).to(device)
                path_input = torch.stack(
                    [p.reshape(24, 2).clone().detach().to(torch.float) for p in path_data]).to(device)
                # scene_input = torch.stack(
                #     [torch.tensor(s.squeeze(0).clone().detach(), dtype=torch.float) for s in scene_np]).to(device)
                scene_input = torch.stack([s.squeeze(0).clone().detach().to(torch.float) for s in scene_np]).to(device)

                data_ori = torch.cat((pose_input.reshape(len(pose_np), -1).to(device), scene_input.reshape(len(pose_np), -1).to(device)), dim=1)
                data_rec = model(pose_input, path_input, scene_input)
                loss_recons = criterion(data_ori, data_rec).cpu().numpy().mean(axis=1)
                for i in range(len(mate[0])):
                    mate_p = mate
                    logits = loss_recons[i]
                    max_loss = max(max_loss, logits.item())
                    abn,note = split_str(mate_p[1][i])
                    frame_start = mate_p[3][i]
                    mate_data = mate_p[1][i]
                    if old_mate_data != mate_data or old_scene!=int(mate_p[0][i]):
                        if num != 0:
                            count_zeros = 0
                            for ii in range(len(gt_tmp) - 1, -1, -1):
                                if gt_tmp[ii] == 0:
                                    count_zeros += 1
                                else:
                                    tp = gt_tmp[ii]
                                    for jj in range(24-sl):
                                        gt_tmp[jj + ii + 1] = tp
                                    break

                            gt_zip = np.concatenate((gt_zip, gt_array))

                            gt_tmp = smooth_scores(gt_tmp.cpu().numpy(),window_size=window_size)

                            pre_zip = np.concatenate((pre_zip, np.array(gt_tmp)))

                        if args.dataset == 'UBnormal':
                            gt_name = f'{abn}_scene_{mate_p[0][i]}_scenario{note}_tracks.txt'
                        else:
                            gt_name = f'{abn}_scene_{mate_p[0][i]}_scenario{note}_tracks.npy'
                        gt_path = os.path.join(gt_dir, gt_name)
                        num, gt_array = get_gt(gt_path)

                        gt_tmp = [0.]*num
                        # 假设 gt_tmp 是 numpy 数组或其他类型数据
                        gt_tmp = torch.tensor(gt_tmp)
                        old_mate_data = mate_data
                        old_scene = int(mate_p[0][i])

                    else:
                        if num>frame_start+sl-1:
                            condition = logits > gt_tmp[frame_start:frame_start + sl]
                            gt_tmp[frame_start:frame_start + sl][condition] = logits.item()

                pbar.update(1)


    # 归一化预测分数
    pre_zip = map_loss_to_score(pre_zip, max_loss, threshold, 0.5)
    auc = roc_auc_score(gt_zip, pre_zip)
    ap = average_precision_score(gt_zip, pre_zip)

    print(f'roc:{auc}\tpr:{ap}')

    return auc, ap


def to_test(checkpoints = '',threshold = 0.5,window_size = 1):
    device = torch.device("cuda:0")  # 将 torch.Tensor 分配到的设备的对象
    _,  _, train_nloader, test_loader = gen_fusion_dataset_dataloader()

    # 检查计算机上的可用 CUDA 设备数量
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("No CUDA devices available on this machine.")

    # 将模型移动到指定的设备
    model = Autoencoder().to(device)

    checkpoint = torch.load(checkpoints)
    model.load_state_dict(checkpoint)
    # checkpoint = torch.load(checkpoints)
    # model.load_state_dict(checkpoint,strict=False)
    for param in model.parameters():
        param.requires_grad = True


    # for i in range(1000):
    #     print(f'window_size:{i + 1}')
    roc,pr = test(test_loader, model, device, threshold,window_size = 12)
    print(f'roc:{roc}\tpr:{pr}')

if __name__ == '__main__':

    to_test(checkpoints = checkpoints,threshold = threshold)

