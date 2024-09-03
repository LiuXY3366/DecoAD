import time

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, average_precision_score
import numpy as np
import os
from tqdm import tqdm

from stage1.dataset import gen_fusion_dataset_dataloader
# from stage1.fusion_no_img import Model
from stage1.fusion import Model


def split_str(str):
    first_part,second_part = str.split("_",1)
    return first_part,second_part

def get_gt(gt_path):
    # 打开二进制文件并读取数据
    with open(gt_path, 'rb') as file:
        # 读取第一行并丢弃
        file.readline()

        # 读取二进制数据
        numpy_data = np.fromfile(file, dtype=np.float64)
        # print(numpy_data)
        numpy_data = 1 - numpy_data
    # numpy_data = np.load(gt_path)

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

# def calculate_rbdc(gt, predictions, threshold=0.5):
#     binary_predictions = (predictions >= threshold).astype(int)
#     true_positives = np.sum((binary_predictions == 1) & (gt == 1))
#     total_predictions = np.sum(binary_predictions)
#     rbdc = true_positives / total_predictions if total_predictions > 0 else 0
#     return rbdc
#
# def calculate_tbdc(gt, predictions, threshold=0.5, min_duration=1):
#     binary_predictions = (predictions >= threshold).astype(int)
#     true_positives = 0
#     current_duration = 0
#     for i in range(len(binary_predictions)):
#         if binary_predictions[i] == 1:
#             current_duration += 1
#             if i == len(binary_predictions) - 1 or binary_predictions[i + 1] == 0:
#                 if current_duration >= min_duration and np.any(gt[i - current_duration + 1:i + 1] == 1):
#                     true_positives += 1
#                 current_duration = 0
#     total_tracks = np.sum(binary_predictions)
#     tbdc = true_positives / total_tracks if total_tracks > 0 else 0
#     return tbdc


def test1(dataloader, model, device,sl=24,threshold=0.5):
    true_num = 0
    n_num = 0
    gt_dir = '/home/liuxinyu/PycharmProjects/KG-VAD/data/UBnormal/gt/'
    old_mate_data = ''
    old_scene = 0
    num = 0
    gt_zip = []
    pre_zip = []
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        with tqdm(total=len(dataloader)) as pbar:
            for pose_np,mate,scene_np,_,path_data  in dataloader:
                for i in range(len(mate[0])):
                    mate_p = mate
                    pose = pose_np[i].reshape(1,2,24,18)
                    pose = pose.to(torch.float).to(device)
                    scene_np_seg = scene_np[i].reshape(1,1,1,512).to(torch.float).to(device)
                    path_data_seg = path_data[i].reshape(-1, 24,2).to(torch.float).to(device)
                    logits = model(pose,path_data_seg,scene_np_seg)
                    # mate_p:[10, 'abnormal__1', 1, 0]   场景编号、信息、人物编号、起始帧
                    abn,note = split_str(mate_p[1][i])
                    frame_start = mate_p[3][i]
                    person = mate_p[2][i]
                    mate_data = mate_p[1][i]
                    if old_mate_data != mate_data or old_scene!=int(mate_p[0][i]):
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
                                    for jj in range(24-sl):
                                        gt_tmp[jj + ii + 1] = tp
                                    break
                            gt_zip = np.concatenate((gt_zip, gt_array))
                            # if gt_name == 'abnormal_scene_3_scenario_1_tracks.txt':
                            #     print(gt_tmp)
                            # pre_zip = np.concatenate((pre_zip, gt_tmp))
                            pre_zip = np.concatenate((pre_zip, gt_tmp))
                        gt_name = f'{abn}_scene_{mate_p[0][i]}_scenario{note}_tracks.txt'
                        # print(gt_name)
                        gt_path = os.path.join(gt_dir, gt_name)
                        num, gt_array = get_gt(gt_path)
                        # print(f'num:{num}')
                        gt_tmp = [0.]*num
                        # gt_tmp = torch.tensor(gt_tmp, dtype=logits.dtype, device=logits.device)
                        old_mate_data = mate_data
                        old_scene = int(mate_p[0][i])

                    else:
                        if num > frame_start:
                            for j in range(sl):
                                if num>frame_start+j and logits>gt_tmp[frame_start+j]:
                                    gt_tmp[frame_start+j] = logits.item()
                pbar.update(1)
    roc_auc_value = roc_auc_score(gt_zip, pre_zip)

    precision, recall, _ = precision_recall_curve(gt_zip, pre_zip)

    # 计算PR-AUC值
    pr_auc_value = auc(recall, precision)

    print(f'ROC:{roc_auc_value}，PR:{pr_auc_value}')

    # 返回PR-AUC值
    return roc_auc_value,pr_auc_value







def test(dataloader, model, device, sl=24, threshold=0.5):
    true_num = 0
    n_num = 0
    gt_dir = '/home/liuxinyu/PycharmProjects/KG-VAD/data/UBnormal/gt/'
    old_mate_data = ''
    old_scene = 0
    num = 0
    gt_zip = []
    pre_zip = []
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        with tqdm(total=len(dataloader)) as pbar:
            print(f'len(dataloader):{len(dataloader)}')
            for pose_np, mate, scene_np, _, path_data in dataloader:
                for i in range(len(mate[0])):
                    mate_p = mate
                    pose = pose_np[i].reshape(1, 2, 24, 18)
                    pose = pose.to(torch.float).to(device)
                    scene_np_seg = scene_np[i].reshape(1, 1, 1, 512).to(torch.float).to(device)
                    path_data_seg = path_data[i].reshape(-1, 24, 2).to(torch.float).to(device)
                    logits = model(pose, path_data_seg, scene_np_seg)

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
                            # if gt_name == 'abnormal_scene_14_scenario_5_tracks.txt':
                            #     print(gt_name)
                            #     print(gt_tmp)
                            #     show_line(gt_tmp, gt_array, gt_name.split('_tracks.txt')[0])
                            # if gt_name == 'abnormal_scene_16_scenario_3_tracks.txt':
                            #     print(gt_name)
                            #     print(gt_tmp)
                            #     show_line(gt_tmp, gt_array, gt_name.split('_tracks.txt')[0])
                            # if gt_name == 'normal_scene_26_scenario_8_tracks.txt':
                            #     print(gt_name)
                            #     print(gt_tmp)
                            #     show_line(gt_tmp, gt_array, gt_name.split('_tracks.txt')[0])
                            # show_line(gt_tmp, gt_array, gt_name.split('_tracks.txt')[0])
                            # pre_zip = np.concatenate((pre_zip, np.array(gt_tmp)))  # 不需要再调用.cpu().numpy()
                            pre_zip = np.concatenate((pre_zip, np.array(gt_tmp.cpu())))

                        gt_name = f'{abn}_scene_{mate_p[0][i]}_scenario{note}_tracks.txt'
                        gt_path = os.path.join(gt_dir, gt_name)
                        num, gt_array = get_gt(gt_path)
                        gt_tmp = torch.zeros(num, dtype=logits.dtype, device=logits.device)  # 初始化为全零张量
                        old_mate_data = mate_data
                        old_scene = int(mate_p[0][i])

                    else:
                        if num > frame_start:
                            start_update_pos = max(num - frame_start, 0)
                            end_update_pos = min(start_update_pos + sl, num)  # 避免超出张量界限

                            # 创建一个与 gt_tmp 切片同形状的布尔张量，初始值为 False
                            condition = torch.zeros(sl, dtype=torch.bool, device=logits.device)

                            # 从 start_update_pos 开始，将条件满足的位置设为 True
                            condition[:end_update_pos - start_update_pos] = logits > gt_tmp[frame_start:frame_start + sl][
                                                                                      :end_update_pos - start_update_pos]

                            # 使用 torch.where 进行条件更新
                            gt_tmp[frame_start:frame_start + sl] = torch.where(condition, logits, gt_tmp[frame_start:frame_start + sl])

                pbar.update(1)
    roc_auc_value = roc_auc_score(gt_zip, pre_zip)

    precision, recall, _ = precision_recall_curve(gt_zip, pre_zip)

    # 计算PR-AUC值
    ap_value = average_precision_score(gt_zip, pre_zip)

    print(f'ROC:{roc_auc_value}，AP:{ap_value}')


    # 返回PR-AUC值
    return roc_auc_value,ap_value

    # return roc_auc_value

    # print(f"测试正确率：{float(true_num)/n_num}")

def show_gt():
    a,b = split_str('normal__10')
    print(a)
    print(b)
    gt_dir = '/home/liuxinyu/PycharmProjects/KG-VAD/data/UBnormal/gt/'
    gt_name = '/home/liuxinyu/PycharmProjects/KG-VAD/data/UBnormal/gt/normal_scene_5_scenario_8_tracks.txt'
    gt_path = os.path.join(gt_dir, gt_name)
    num, gt_array = get_gt(gt_path)
    print(gt_array)
from torchsummary import summary
def traverse():
    device = torch.device("cuda:0")  # 将 torch.Tensor 分配到的设备的对象
    _, _, _, train_nloader, train_aloader, test_loader = gen_fusion_dataset_dataloader()
    model = Model().to(device)

    checkpoints = '/home/liuxinyu/PycharmProjects/KG-VAD/000000000000000000pkl/model0.80429.pkl'
    checkpoint = torch.load(checkpoints)
    start_time = time.perf_counter()
    model.load_state_dict(checkpoint)
    auc = test(test_loader, model, device)
    cost_time = time.perf_counter() - start_time
    print(np.sum(cost_time))   # 3691.3083049989655
    print("FPS is: ", 92640 / (np.sum(cost_time) + 3691.3083049989655))
    # print("FPS is: ", test_loader.size / np.sum(cost_time))
    # summary(model, (3, 224, 224))
    # summary(model, [(1, 2, 24, 18), (-1, 24, 2), (1, 1, 1, 512)])
    print(f'auc:{auc}')

# sl=24	auc:0.7956781463748512
#
if __name__ == '__main__':
      # traverse()
      print((1176.*0.22212370595661923)/92640.)
      print((1176. * 5.62501372769475e-06) / 92640.)

      print((1176. * 0.0003350840415805578) / 92640.)

      print((1176. * 0.000207193021196872) / 92640.)