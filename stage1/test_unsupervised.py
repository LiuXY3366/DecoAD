import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score
import numpy as np
import os

from torch import nn
from tqdm import tqdm

from stage1.dataset import gen_fusion_dataset_dataloader


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

        return numpy_data.shape[0],numpy_data
def get_gt_SH(gt_path):
    numpy_data = np.load(gt_path)
    return numpy_data.shape[0],numpy_data

def get_gt_NWPUC(gt_path):
    numpy_data = np.load(gt_path)
    return numpy_data.shape[0],numpy_data


# def map_loss_to_score(loss_arr,max_loss, threshold,thr = 0.5):
#     for i in range(len(loss_arr)):
#         loss = loss_arr[i]
#         if loss <= threshold:
#             # 映射到0到0.5之间
#             loss_arr[i] = thr * (loss / threshold)
#         else:
#             # 映射到0.5到1之间
#             loss_arr[i] = thr + thr * ((loss - threshold) / (max_loss - threshold))
#     return loss_arr

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


def test(dataloader, model, device,threshold,sl=24,dataset_name = 'NWPUC'):
    true_num = 0
    n_num = 0
    gt_dir = f'/home/liuxinyu/PycharmProjects/KG-VAD-STC/data/{dataset_name}/gt/'
    old_mate_data = ''
    old_scene = 0
    num = 0
    gt_zip = []
    pre_zip = []
    max_loss = 0.
    with torch.no_grad():
        criterion = nn.MSELoss()
        model.eval()
        pred = torch.zeros(0)
        with tqdm(total=len(dataloader)) as pbar:
            for pose_np,mate,scene_np,_  in dataloader:
                for i in range(len(mate[0])):
                    mate_p = mate
                    pose = pose_np[i].reshape(1,2,24,18)
                    pose = pose.to(torch.float).to(device)
                    scene_np_seg = scene_np[i]
                    scene_np_seg = scene_np_seg.to(torch.float).to(device)
                    batch_size = pose.size(0)
                    data_ori = torch.cat((pose.view(batch_size, -1), scene_np_seg.view(batch_size, -1)), dim=1)
                    data_rec = model(pose,scene_np_seg)
                    logits = criterion(data_ori, data_rec)
                    max_loss = max(max_loss, logits.item())
                    abn,note = split_str(mate_p[1][i])
                    frame_start = mate_p[3][i]
                    person = mate_p[2][i]
                    mate_data = mate_p[1][i]
                    if old_mate_data != mate_data or old_scene!=int(mate_p[0][i]):
                        if num != 0:
                            count_zeros = 0
                            tp = 0
                            for ii in range(len(gt_tmp) - 1, -1, -1):
                                if gt_tmp[ii] == 0:
                                    count_zeros += 1
                                else:
                                    tp = gt_tmp[ii]
                                    for jj in range(24-sl):
                                        gt_tmp[jj + ii + 1] = tp
                                    break

                            gt_zip = np.concatenate((gt_zip, gt_array))
                            pre_zip = np.concatenate((pre_zip, np.array(gt_tmp.cpu().numpy())))

                        if dataset_name == 'UBnormal':
                            gt_name = f'{abn}_scene_{mate_p[0][i]}_scenario{note}_tracks.txt'
                            gt_path = os.path.join(gt_dir, gt_name)
                            num, gt_array = get_gt(gt_path)
                        elif dataset_name == 'NWPUC':
                            # print(int(mate_p[0][i]))
                            note_clean = note.lstrip('_')
                            note_int = int(note_clean)
                            # print(note_int)
                            gt_name = "D{:03}_{:02}.npy".format(int(mate_p[0][i]), note_int)
                            # print(gt_name)
                            # print(gt_name)
                            gt_path = os.path.join(gt_dir, gt_name)
                            num, gt_array = get_gt_NWPUC(gt_path)
                        elif dataset_name == 'ShanghaiTech':
                            gt_name = f'{abn}_scene_{mate_p[0][i]}_scenario{note}_tracks.npy'
                            gt_path = os.path.join(gt_dir, gt_name)
                            num, gt_array = get_gt_SH(gt_path)
                        # print(f'num:{num}')
                        gt_tmp = [0.]*num
                        gt_tmp = torch.tensor(gt_tmp, dtype=logits.dtype, device=logits.device)
                        old_mate_data = mate_data
                        old_scene = int(mate_p[0][i])

                    else:
                        if num>frame_start+sl-1:
                            condition = logits > gt_tmp[frame_start:frame_start + sl]
                            gt_tmp[frame_start:frame_start + sl][condition] = logits.item()


                            # for j in range(sl):
                            #     if logits>gt_tmp[frame_start+j]:
                            #         gt_tmp[frame_start+j] = logits.item()
                pbar.update(1)

    roc_auc_value = 0

    pre_zip_new = map_loss_to_score(pre_zip, max_loss, threshold, 0.5)
    roc_auc_value = roc_auc_score(gt_zip, pre_zip_new)
    precision, recall, _ = precision_recall_curve(gt_zip, pre_zip_new)
    pr_auc_value = auc(recall, precision)
    ap_value = average_precision_score(gt_zip, pre_zip_new)  # 计算平均精度 (AP)
    print(f'ROC:{roc_auc_value}，PR:{pr_auc_value}，AP:{ap_value}')

    return roc_auc_value

def show_gt():
    a,b = split_str('normal__10')
    print(a)
    print(b)
    gt_dir = '/home/liuxinyu/PycharmProjects/KG-VAD-STC/data/UBnormal/gt/'
    gt_name = '/home/liuxinyu/PycharmProjects/KG-VAD-STC/data/UBnormal/gt/normal_scene_5_scenario_8_tracks.txt'
    gt_path = os.path.join(gt_dir, gt_name)
    num, gt_array = get_gt(gt_path)
    print(gt_array)

# def traverse():
#     device = torch.device("cuda:0")  # 将 torch.Tensor 分配到的设备的对象
#     _, _, _, train_nloader, train_aloader, test_loader = gen_fusion_dataset_dataloader()
#     model = Model().to(device)
#
#     checkpoints = '/home/liuxinyu/PycharmProjects/KG-VAD-STC/ckpt1/model0.78699.pkl'
#     checkpoint = torch.load(checkpoints)
#
#     model.load_state_dict(checkpoint)
#     for param in model.parameters():
#         param.requires_grad = True
#
#     # auc = test(test_loader, model, device, sl=24)
#     # print(f'sl={24}\tauc:{auc}')
#
#     # for i in range(25):
#     auc = test(test_loader, model, device)
#     print(f'auc:{auc}')

# sl=24	auc:0.7956781463748512

# if __name__ == '__main__':
#     traverse()


