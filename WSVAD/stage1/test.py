import time

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, average_precision_score
import numpy as np
import os
from tqdm import tqdm

from WSVAD.stage1.dataset import gen_fusion_dataset_dataloader
# from stage1.fusion_no_img import Model
from WSVAD.stage1.fusion import Model
from WSVAD.stage1.args import init_parser

# 初始化解析器
# from WSVAD.stage1.test import gaussian_smooth

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

def calculate_rbdc(gt, predictions, threshold=0.5):
    binary_predictions = [1 if p >= threshold else 0 for p in predictions]

    tp = sum(1 for gt_val, pred_val in zip(gt, binary_predictions) if gt_val == pred_val == 1)
    fp = sum(1 for gt_val, pred_val in zip(gt, binary_predictions) if gt_val == 0 and pred_val == 1)
    fn = sum(1 for gt_val, pred_val in zip(gt, binary_predictions) if gt_val == 1 and pred_val == 0)

    rbdc_score = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return rbdc_score


def calculate_tbdc(gt, predictions, threshold=0.5, track_threshold=1):
    binary_predictions = [1 if p >= threshold else 0 for p in predictions]

    # 定义一个函数来识别和评估异常轨迹
    def evaluate_tracks(binary_preds, gt):
        true_tracks = 0
        current_track_length = 0
        for pred, actual in zip(binary_preds, gt):
            if pred == 1:
                current_track_length += 1
                if actual == 1 and current_track_length >= track_threshold:
                    true_tracks += 1
                    current_track_length = 0
            else:
                current_track_length = 0
        return true_tracks

    total_tracks = sum(1 for pred in binary_predictions if pred == 1)
    true_tracks = evaluate_tracks(binary_predictions, gt)

    tbdc_score = true_tracks / total_tracks if total_tracks > 0 else 0
    return tbdc_score


# def scores_smooth(gt_zip,pre_zip,sigma = 1000):
#     # 归一化预测分数
#     pre_zip = np.clip(pre_zip, 0, 1)
#     max_auc = 0.
#     max_sigma =0
#     max_ap = 0.
#     for idxx in range(sigma):
#         pre_zip = gaussian_smooth(pre_zip, sigma=idxx+1)
#         # 计算 ROC-AUC 和 AP
#         roc_auc_value = roc_auc_score(gt_zip, pre_zip)
#         ap_value = average_precision_score(gt_zip, pre_zip)
#         if max_auc<roc_auc_value:
#             max_auc = roc_auc_value
#             max_sigma = idxx+1
#             max_ap = ap_value
#
#         # 绘制 Precision-Recall 曲线
#         precision, recall, _ = precision_recall_curve(gt_zip, pre_zip)
#
#         # 输出结果
#     print(f"ROC-AUC: {max_auc:.4f}, AP: {max_ap:.4f}, sigma:{max_sigma}")
#     return max_auc, max_ap


# roc:0.7536008850137821	pr:0.10788295546578325	windows:157

def smooth_scores(scores, window_size=12):
    if window_size > len(scores):
        return scores
    # 检查窗口大小
    if window_size < 1:
        raise ValueError("Window size must be greater than or equal to 1.")

    # 使用滑动窗口平滑分数
    smoothed_scores = np.convolve(scores, np.ones(window_size) / window_size, mode='same')

    return smoothed_scores


def test(dataloader, model, device, sl=24, threshold=0.5 ,window_size=156):
    true_num = 0
    n_num = 0
    gt_dir = f'{args.data_dir}{args.dataset}/gt/'
    old_mate_data = ''
    old_scene = 0
    num = 0
    gt_zip = []
    pre_zip = []
    with torch.no_grad():
        model.eval()

        pred = torch.zeros(0)
        with tqdm(total=len(dataloader)) as pbar:
            for pose_np, mate, scene_np, _, path_data in dataloader:
                pose_input = torch.stack(
                    [d.clone().detach().to(dtype=torch.float) for d in pose_np]
                ).to(device)

                path_input = torch.stack(
                    [p.reshape(24, 2).clone().detach().to(dtype=torch.float) for p in path_data]
                ).to(device)

                scene_input = torch.stack(
                    [s.squeeze(0).clone().detach().to(dtype=torch.float) for s in scene_np]
                ).to(device)
                logits_dir = model(pose_input, path_input, scene_input)
                for i in range(len(mate[0])):
                    mate_p = mate
                    logits = logits_dir[i]

                    # mate_p:[10, 'abnormal__1', 1, 0]   场景编号、信息、人物编号、起始帧
                    abn, note = split_str(mate_p[1][i])
                    frame_start = mate_p[3][i]
                    person = mate_p[2][i]
                    mate_data = mate_p[1][i]
                    if old_mate_data != mate_data or old_scene != int(mate_p[0][i]):
                        # 检测异常
                        if num != 0:
                            count_zeros = 0
                            tp = 0
                            # 从数组的最后一个元素开始往前遍历
                            for ii in range(len(gt_tmp) - 1, -1, -1):
                                if gt_tmp[ii] == 0:
                                    count_zeros += 1
                                else:
                                    tp = gt_tmp[ii]
                                    for jj in range(24 - sl):
                                        gt_tmp[jj + ii + 1] = tp
                                    break
                            gt_zip = np.concatenate((gt_zip, gt_array))
                            gt_tmp = smooth_scores(gt_tmp.cpu(),window_size)
                            # print(f'old_mate_data:{old_mate_data}\told_scene:{old_scene}')
                            # if (old_mate_data.split('_')[-1] in ['1', '2', '3', '7', '6', '11', '12', '13', '14', '18', '19', '20']) and old_scene == 235:
                            #     print(gt_tmp)
                            #     if not isinstance(gt_tmp, np.ndarray):
                            #         gt_tmp = np.array(gt_tmp)  # 转换为 NumPy 数组
                            #
                            #     # 保存到 .npy 文件
                            #     np.save(f"nwpuc_235_{old_mate_data.split('_')[-1]}.npy", gt_tmp)
                            pre_zip = np.concatenate((pre_zip, np.array(gt_tmp)))

                        if args.dataset == 'UBnormal':
                            gt_name = f'{abn}_scene_{mate_p[0][i]}_scenario{note}_tracks.txt'
                        else:
                            gt_name = f'{abn}_scene_{mate_p[0][i]}_scenario{note}_tracks.npy'
                        gt_path = os.path.join(gt_dir, gt_name)
                        num, gt_array = get_gt(gt_path)
                        gt_tmp = torch.zeros(num, dtype=logits.dtype, device=logits.device)  # 初始化为全零张量
                        old_mate_data = mate_data
                        old_scene = int(mate_p[0][i])

                    # 更新 gt_tmp 张量
                    else:
                        if num > frame_start:
                            # 确保 start 和 end 不超出边界
                            start_update_pos = max(0, frame_start)
                            end_update_pos = min(frame_start + sl, num)

                            # 确保 condition 和 logits 的形状匹配
                            logits_slice = logits[:end_update_pos - start_update_pos]
                            gt_slice = gt_tmp[frame_start:end_update_pos]

                            # 创建布尔条件张量，并更新 gt_tmp
                            condition = logits_slice > gt_slice
                            gt_tmp[frame_start:end_update_pos] = torch.where(condition, logits_slice, gt_slice)
                pbar.update(1)

    auc = roc_auc_score(gt_zip, pre_zip)
    ap = average_precision_score(gt_zip, pre_zip)

    # max_auc, max_ap = scores_smooth(gt_zip,pre_zip)
    print(f'auc:{auc}\tap:{ap}')
    return auc, ap



def to_test(checkpoints = ''):
    max_auc = 0.
    device = torch.device("cuda:0")  # 将 torch.Tensor 分配到的设备的对象
    _, _, _, train_nloader, train_aloader, test_loader = gen_fusion_dataset_dataloader()

    # 检查计算机上的可用 CUDA 设备数量
    device_count = torch.cuda.device_count()
    if device_count < 1:
        raise RuntimeError("No CUDA devices available on this machine.")

    model = Model().to(device)

    checkpoint = torch.load(checkpoints)

    model.load_state_dict(checkpoint)

    for param in model.parameters():
        param.requires_grad = True

    for i in range (100):
        roc, pr = test(test_loader, model, device,window_size=6*(i+1))

        print(f'roc:{roc}\tpr:{pr}\twindows:{6*(i+1)}')



if __name__ == '__main__':
    to_test('')