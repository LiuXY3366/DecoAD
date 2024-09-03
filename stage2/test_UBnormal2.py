import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, average_precision_score
import numpy as np
import os
from tqdm import tqdm


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
    # numpy_data = np.load(gt_path)  # nwpuc STC

    return numpy_data.shape[0],numpy_data




# def test(dataloader, model, device,sl = 24):
#     true_num = 0
#     n_num = 0
#     gt_dir = '/home/liuxinyu/PycharmProjects/KG-VAD/data/UBnormal/gt/'
#     old_mate_data = ''
#     old_scene = 0
#     num = 0
#     gt_zip = []
#     pre_zip = []
#     with torch.no_grad():
#         model.eval()
#         pred = torch.zeros(0)
#         with tqdm(total=len(dataloader)) as pbar:
#             for pose_np,mate,scene_np,_ ,path_data in dataloader:
#                 for i in range(len(mate[0])):
#                     mate_p = mate
#                     pose = pose_np[i].reshape(1,2,24,18)
#                     pose = pose.to(torch.float).to(device)
#                     scene_np_seg = scene_np[i].reshape(1,1,1,512).to(torch.float).to(device)
#                     path_data_seg = path_data[i].reshape(-1, 24,2).to(torch.float).to(device)
#                     logits = model(pose,path_data_seg,scene_np_seg)
#                     # mate_p:[10, 'abnormal__1', 1, 0]   场景编号、信息、人物编号、起始帧
#                     abn,note = split_str(mate_p[1][i])
#                     frame_start = mate_p[3][i]
#                     person = mate_p[2][i]
#                     mate_data = mate_p[1][i]
#                     if old_mate_data != mate_data or old_scene!=int(mate_p[0][i]):
#                         # 检测异常
#                         if num != 0:
#                             count_zeros = 0
#                             tp = 0
#                             # 从数组的最后一个元素开始往前遍历
#                             for ii in range(len(gt_tmp) - 1, -1, -1):
#                                 if gt_tmp[ii] == 0:
#                                     count_zeros += 1
#                                 else:
#                                     tp = gt_tmp[ii]
#                                     for jj in range(24-sl):
#                                         gt_tmp[jj + ii + 1] = tp
#                                     break
#                             gt_zip = np.concatenate((gt_zip, gt_array))
#                             pre_zip = np.concatenate((pre_zip, gt_tmp))
#                         gt_name = f'{abn}_scene_{mate_p[0][i]}_scenario{note}_tracks.txt'
#                         gt_path = os.path.join(gt_dir, gt_name)
#                         num, gt_array = get_gt(gt_path)
#                         # print(f'num:{num}')
#                         gt_tmp = [0.]*num
#                         old_mate_data = mate_data
#                         old_scene = int(mate_p[0][i])
#                     else:
#                         if num>frame_start:
#                             for j in range(sl):
#                                 if num>frame_start+j and logits>gt_tmp[frame_start+j]:
#                                     gt_tmp[frame_start+j] = logits.item()
#
#                 pbar.update(1)
#     roc_auc_value = roc_auc_score(gt_zip, pre_zip)
#
#     precision, recall, _ = precision_recall_curve(gt_zip, pre_zip)
#
#     # 计算PR-AUC值
#     ap_value = average_precision_score(gt_zip, pre_zip)
#
#     print(f'ROC:{roc_auc_value}，PR:{ap_value}')
#
#
#     auc_value = roc_auc_score(gt_zip, pre_zip)
#     return auc_value

    # print(f"测试正确率：{float(true_num)/n_num}")
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
    return roc_auc_value

def show_gt():
    a,b = split_str('normal__10')
    print(a)
    print(b)
    gt_dir = '/home/liuxinyu/PycharmProjects/KG-VAD/data/UBnormal/gt/'
    gt_name = '/home/liuxinyu/PycharmProjects/KG-VAD/data/UBnormal/gt/normal_scene_5_scenario_8_tracks.txt'
    gt_path = os.path.join(gt_dir, gt_name)
    num, gt_array = get_gt(gt_path)
    print(gt_array)




if __name__ == '__main__':
    show_gt()


