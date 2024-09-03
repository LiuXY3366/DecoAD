import json
import math
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

SHANGHAITECH_HR_SKIP = [(1, 130), (1, 135), (1, 136), (6, 144), (6, 145), (12, 152)]


def normalize_pose(pose_data, **kwargs):
    """
    Normalize keypoint values to the range of [-1, 1]
    :param pose_data: Formatted as [N, T, V, F], e.g. (Batch=64, Frames=12, 18, 3)
    :param vid_res:
    :param symm_range:
    :return:
    """
    vid_res = kwargs.get('vid_res', [856, 480])
    symm_range = kwargs.get('symm_range', False)
    # sub_mean = kwargs.get('sub_mean', True)
    # scale = kwargs.get('scale', False)
    # scale_proportional = kwargs.get('scale_proportional', True)

    vid_res_wconf = vid_res + [1]
    norm_factor = np.array(vid_res_wconf)
    pose_data_normalized = pose_data / norm_factor
    pose_data_centered = pose_data_normalized
    if symm_range:  # Means shift data to [-1, 1] range
        pose_data_centered[..., :2] = 2 * pose_data_centered[..., :2] - 1

    pose_data_zero_mean = pose_data_centered
    # return pose_data_zero_mean

    pose_data_zero_mean[..., :2] = (pose_data_centered[..., :2] - pose_data_centered[..., :2].mean(axis=(1, 2))[:, None, None, :]) / pose_data_centered[..., 1].std(axis=(1, 2))[:, None, None, None]
    return pose_data_zero_mean

def get_ab_labels_weakly(global_data_np_ab, segs_meta_ab, path_to_vid_dir='', segs_root=''):
    pose_segs_root = segs_root
    clip_list = os.listdir(pose_segs_root)
    clip_list = sorted(
        fn.replace("alphapose_tracked_person.json", "annotations") for fn in clip_list if fn.endswith('.json'))
    labels = np.ones_like(global_data_np_ab)  # 默认所有帧标记为正常

    for clip in tqdm(clip_list):
        type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_annotations.*', clip)[0]
        if type == "normal":
            continue

        # 更新标签，将所有帧标记为异常
        # print('===============================================')
        # print(labels[(type == 'abnormal') & (segs_meta_ab[:, 1] == clip_id) & (segs_meta_ab[:, 0] == scene_id)])
        labels[(type=='abnormal')] = -1
        # print(labels[(type=='abnormal')&(segs_meta_ab[:, 1] == clip_id)&(segs_meta_ab[:, 0] == scene_id)])
    # print('===============================================')
    # print(labels)
    return labels[:, 0, 0, 0]

def get_ab_labels(global_data_np_ab, segs_meta_ab, path_to_vid_dir='', segs_root=''):
    pose_segs_root = segs_root
    clip_list = os.listdir(pose_segs_root)
    clip_list = sorted(
        fn.replace("alphapose_tracked_person.json", "annotations") for fn in clip_list if fn.endswith('.json'))
    labels = np.ones_like(global_data_np_ab)
    for clip in tqdm(clip_list):
        type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_annotations.*', clip)[0]
        if type == "normal":
            continue
        clip_id = type + "_" + clip_id
        clip_metadata_inds = np.where((segs_meta_ab[:, 1] == clip_id) &
                                      (segs_meta_ab[:, 0] == scene_id))[0]
        clip_metadata = segs_meta_ab[clip_metadata_inds]
        clip_res_fn = os.path.join(path_to_vid_dir, "Scene{}".format(scene_id), clip)
        filelist = sorted(os.listdir(clip_res_fn))
        clip_gt_lst = [np.array(Image.open(os.path.join(clip_res_fn, fname)).convert('L')) for fname in filelist if fname.endswith(('.jpg', '.png', '.bmp', '.jpeg'))]
        # clip_gt_lst = [np.array(Image.open(os.path.join(clip_res_fn, fname)).convert('L')) for fname in filelist]
        # FIX shape bug
        clip_shapes = set([clip_gt.shape for clip_gt in clip_gt_lst])
        min_width = min([clip_shape[0] for clip_shape in clip_shapes])
        min_height = min([clip_shape[1] for clip_shape in clip_shapes])
        clip_labels = np.array([clip_gt[:min_width, :min_height] for clip_gt in clip_gt_lst])
        gt_file = os.path.join("/home/liuxinyu/PycharmProjects/KG-VAD/data/UBnormal/gt", clip.replace("annotations", "tracks.txt"))
        clip_gt = np.zeros_like(clip_labels)
        with open(gt_file) as f:
            abnormality = f.readlines()
            for ab in abnormality:
                i, start, end = ab.strip("\n").split(",")
                for t in range(int(float(start)), int(float(end))):
                    clip_gt[t][clip_labels[t] == int(float(i))] = 1
        for t in range(clip_gt.shape[0]):
            if (clip_gt[t] != 0).any():  # Has abnormal event
                ab_metadata_inds = np.where(clip_metadata[:, 3].astype(int) == t)[0]
                # seg = clip_segs[ab_metadata_inds][:, :2, 0]
                clip_fig_idxs = set([arr[2] for arr in segs_meta_ab[ab_metadata_inds]])
                for person_id in clip_fig_idxs:
                    person_metadata_inds = np.where((segs_meta_ab[:, 1] == clip_id) &
                                                    (segs_meta_ab[:, 0] == scene_id) &
                                                    (segs_meta_ab[:, 2] == person_id) &
                                                    (segs_meta_ab[:, 3].astype(int) == t))[0]
                    data = np.floor(global_data_np_ab[person_metadata_inds].T).astype(int)
                    if data.shape[-1] != 0:
                        if clip_gt[t][
                            np.clip(data[:, 0, 1], 0, clip_gt.shape[1] - 1),
                            np.clip(data[:, 0, 0], 0, clip_gt.shape[2] - 1)
                        ].sum() > data.shape[0] / 2:
                            # This pose is abnormal
                            labels[person_metadata_inds] = -1
    return labels[:, 0, 0, 0]


# def get_ab_labels(global_data_np_ab, segs_meta_ab, path_to_vid_dir='', segs_root=''):
#     pose_segs_root = segs_root
#     clip_list = os.listdir(pose_segs_root)
#     clip_list = sorted(
#         fn.replace("alphapose_tracked_person.json", "annotations") for fn in clip_list if fn.endswith('.json'))
#     labels = np.ones_like(global_data_np_ab)
#     # print(labels.shape)   (26228, 3, 24, 18)
#     for clip in tqdm(clip_list):
#         type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_annotations.*', clip)[0]
#         if type == "normal":
#             continue
#         clip_id = type + "_" + clip_id
#         clip_metadata_inds = np.where((segs_meta_ab[:, 1] == clip_id) &
#                                       (segs_meta_ab[:, 0] == scene_id))[0]
#         clip_metadata = segs_meta_ab[clip_metadata_inds]
#         clip_res_fn = os.path.join(path_to_vid_dir, "Scene{}".format(scene_id), clip)
#         filelist = sorted(os.listdir(clip_res_fn))
#         clip_gt_lst = [np.array(Image.open(os.path.join(clip_res_fn, fname)).convert('L')) for fname in filelist if fname.endswith(('.jpg', '.png', '.bmp', '.jpeg'))]
#         # clip_gt_lst = [np.array(Image.open(os.path.join(clip_res_fn, fname)).convert('L')) for fname in filelist]
#         # FIX shape bug
#         clip_shapes = set([clip_gt.shape for clip_gt in clip_gt_lst])
#         min_width = min([clip_shape[0] for clip_shape in clip_shapes])
#         min_height = min([clip_shape[1] for clip_shape in clip_shapes])
#         clip_labels = np.array([clip_gt[:min_width, :min_height] for clip_gt in clip_gt_lst])
#         # gt_file = os.path.join("/home/liuxinyu/PycharmProjects/KG-VAD/data/UBnormal/gt", clip.replace("annotations", "tracks.txt"))
#         gt_file = os.path.join("/home/liuxinyu/PycharmProjects/KG-VAD/data/UBnormal/gt", clip.replace("annotations", "tracks.txt"))
#         clip_gt = np.zeros_like(clip_labels)
#         with open(gt_file) as f:
#             abnormality = f.readlines()
#             for ab in abnormality:
#                 i, start, end = ab.strip("\n").split(",")
#                 for t in range(int(float(start)), int(float(end))):
#                     clip_gt[t][clip_labels[t] == int(float(i))] = 1
#
#         for t in range(clip_gt.shape[0]):
#             if (clip_gt[t] != 0).any():  # Has abnormal event
#                 ab_metadata_inds = np.where(clip_metadata[:, 3].astype(int) == t)[0]
#                 # seg = clip_segs[ab_metadata_inds][:, :2, 0]
#                 clip_fig_idxs = set([arr[2] for arr in segs_meta_ab[ab_metadata_inds]])
#                 for person_id in clip_fig_idxs:
#                     person_metadata_inds = np.where((segs_meta_ab[:, 1] == clip_id) &
#                                                     (segs_meta_ab[:, 0] == scene_id) &
#                                                     (segs_meta_ab[:, 2] == person_id) &
#                                                     (segs_meta_ab[:, 3].astype(int) == t))[0]
#                     data = np.floor(global_data_np_ab[person_metadata_inds].T).astype(int)
#                     if data.shape[-1] != 0:
#                         if clip_gt[t][
#                             np.clip(data[:, 0, 1], 0, clip_gt.shape[1] - 1),
#                             np.clip(data[:, 0, 0], 0, clip_gt.shape[2] - 1)
#                         ].sum() > data.shape[0] / 2:
#                             # This pose is abnormal
#                             labels[person_metadata_inds] = -1
#     return labels[:, 0, 0, 0]

'''
    clip_segs_data_np, clip_segs_meta, clip_keys, single_pos_np, _, score_segs_data_np = gen_clip_seg_data_np(
        clip_dict, start_ofst,
        seg_stride,
        seg_len,
        scene_id=scene_id,
        clip_id=clip_id,
        ret_keys=ret_keys,
        dataset=dataset)
'''

def gen_clip_seg_data_np(clip_dict, start_ofst=0, seg_stride=4, seg_len=24, scene_id='', clip_id='', ret_keys=False,
                         global_pose_data=[], dataset="UBnormal"):
    """
    Generate an array of segmented sequences, each object is a segment and a corresponding metadata array
    """
    pose_segs_data = []
    score_segs_data = []
    pose_segs_meta = []
    person_keys = {}
    for idx in sorted(clip_dict.keys(), key=lambda x: int(x)):
        sing_pose_np, sing_pose_meta, sing_pose_keys, sing_scores_np = single_pose_dict2np(clip_dict, idx)
        if dataset == "UBnormal":
            key = ('{:02d}_{}_{:02d}'.format(int(scene_id), clip_id, int(idx)))
        else:
            key = ('{:02d}_{:04d}_{:02d}'.format(int(scene_id), int(clip_id), int(idx)))
        person_keys[key] = sing_pose_keys
        curr_pose_segs_np, curr_pose_segs_meta, curr_pose_score_np = split_pose_to_segments(sing_pose_np,
                                                                                            sing_pose_meta,
                                                                                            sing_pose_keys,
                                                                                            start_ofst, seg_stride,
                                                                                            seg_len,
                                                                                            scene_id=scene_id,
                                                                                            clip_id=clip_id,
                                                                                            single_score_np=sing_scores_np,
                                                                                            dataset=dataset)
        pose_segs_data.append(curr_pose_segs_np)
        score_segs_data.append(curr_pose_score_np)
        if sing_pose_np.shape[0] > seg_len:
            global_pose_data.append(sing_pose_np)
        pose_segs_meta += curr_pose_segs_meta
    if len(pose_segs_data) == 0:
        # lll
        pose_segs_data_np = np.empty(0).reshape(0, seg_len, 17, 3)
        # pose_segs_data_np = np.empty(0).reshape(0, seg_len, 17, 2)
        score_segs_data_np = np.empty(0).reshape(0, seg_len)
    else:
        pose_segs_data_np = np.concatenate(pose_segs_data, axis=0)
        score_segs_data_np = np.concatenate(score_segs_data, axis=0)
    global_pose_data_np = np.concatenate(global_pose_data, axis=0)
    del pose_segs_data
    # del global_pose_data
    if ret_keys:
        return pose_segs_data_np, pose_segs_meta, person_keys, global_pose_data_np, global_pose_data, score_segs_data_np
    else:
        return pose_segs_data_np, pose_segs_meta, global_pose_data_np, global_pose_data, score_segs_data_np

def single_pose_dict2np(person_dict, idx):
    single_person = person_dict[str(idx)]
    sing_pose_np = []
    sing_scores_np = []
    if isinstance(single_person, list):
        single_person_dict = {}
        for sub_dict in single_person:
            single_person_dict.update(**sub_dict)
        single_person = single_person_dict
    single_person_dict_keys = sorted(single_person.keys())
    sing_pose_meta = [int(idx), int(single_person_dict_keys[0])]  # Meta is [index, first_frame]
    for key in single_person_dict_keys:
        curr_pose_np = np.array(single_person[key]['keypoints']).reshape(-1, 3)
        sing_pose_np.append(curr_pose_np)
        sing_scores_np.append(single_person[key]['scores'])
    sing_pose_np = np.stack(sing_pose_np, axis=0)
    sing_scores_np = np.stack(sing_scores_np, axis=0)
    return sing_pose_np, sing_pose_meta, single_person_dict_keys, sing_scores_np


def is_single_person_dict_continuous(sing_person_dict):
    """
    Checks if an input clip is continuous or if there are frames missing
    :return:
    """
    start_key = min(sing_person_dict.keys())
    person_dict_items = len(sing_person_dict.keys())
    sorted_seg_keys = sorted(sing_person_dict.keys(), key=lambda x: int(x))
    return is_seg_continuous(sorted_seg_keys, start_key, person_dict_items)


def is_seg_continuous(sorted_seg_keys, start_key, seg_len, missing_th=2):
    """
    Checks if an input clip is continuous or if there are frames missing
    :param sorted_seg_keys:
    :param start_key:
    :param seg_len:
    :param missing_th: The number of frames that are allowed to be missing on a sequence,
    i.e. if missing_th = 1 then a seg for which a single frame is missing is considered continuous
    :return:
    """
    start_idx = sorted_seg_keys.index(start_key)
    expected_idxs = list(range(start_key, start_key + seg_len))
    act_idxs = sorted_seg_keys[start_idx: start_idx + seg_len]
    min_overlap = seg_len - missing_th
    key_overlap = len(set(act_idxs).intersection(expected_idxs))
    if key_overlap >= min_overlap:
        return True
    else:
        return False


def split_pose_to_segments(single_pose_np, single_pose_meta, single_pose_keys, start_ofst=0, seg_dist=6, seg_len=24,
                           scene_id='', clip_id='', single_score_np=None, dataset="UBnormal"):
    clip_t, kp_count, kp_dim = single_pose_np.shape
    pose_segs_np = np.empty([0, seg_len, kp_count, kp_dim])
    pose_score_np = np.empty([0, seg_len])
    pose_segs_meta = []
    num_segs = np.ceil((clip_t - seg_len) / seg_dist).astype(np.int64)
    single_pose_keys_sorted = sorted([int(i) for i in single_pose_keys])  # , key=lambda x: int(x))
    for seg_ind in range(num_segs):
        start_ind = start_ofst + seg_ind * seg_dist
        start_key = single_pose_keys_sorted[start_ind]
        if is_seg_continuous(single_pose_keys_sorted, start_key, seg_len):
            curr_segment = single_pose_np[start_ind:start_ind + seg_len].reshape(1, seg_len, kp_count, kp_dim)
            curr_score = single_score_np[start_ind:start_ind + seg_len].reshape(1, seg_len)
            pose_segs_np = np.append(pose_segs_np, curr_segment, axis=0)
            pose_score_np = np.append(pose_score_np, curr_score, axis=0)
            if dataset == "UBnormal":
                pose_segs_meta.append([int(scene_id), clip_id, int(single_pose_meta[0]), int(start_key)])
            else:
                pose_segs_meta.append([int(scene_id), int(clip_id), int(single_pose_meta[0]), int(start_key)])

    return pose_segs_np, pose_segs_meta, pose_score_np

def get_aff_trans_mat(sx=1, sy=1, tx=0, ty=0, rot=0, shearx=0., sheary=0., flip=False):
    """
    Generate affine transfomation matrix (torch.tensor type) for transforming pose sequences
    :rot is given in degrees
    """
    cos_r = math.cos(math.radians(rot))
    sin_r = math.sin(math.radians(rot))
    flip_mat = torch.eye(3, dtype=torch.float32)
    if flip:
        flip_mat[0, 0] = -1.0
    trans_scale_mat = torch.tensor([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], dtype=torch.float32)
    shear_mat = torch.tensor([[1, shearx, 0], [sheary, 1, 0], [0, 0, 1]], dtype=torch.float32)
    rot_mat = torch.tensor([[cos_r, -sin_r, 0], [sin_r, cos_r, 0], [0, 0, 1]], dtype=torch.float32)
    aff_mat = torch.matmul(rot_mat, trans_scale_mat)
    aff_mat = torch.matmul(shear_mat, aff_mat)
    aff_mat = torch.matmul(flip_mat, aff_mat)
    return aff_mat

def apply_pose_transform(pose, trans_mat):
    """ Given a set of pose sequences of shape (Channels, Time_steps, Vertices, M[=num of figures])
    return its transformed form of the same sequence. 3 Channels are assumed (x, y, conf) """

    # We isolate the confidence vector, replace with ones, than plug back after transformation is done
    conf = np.expand_dims(pose[2], axis=0)
    ones_vec = np.ones_like(conf)
    pose_w_ones = np.concatenate([pose[:2], ones_vec], axis=0)
    if len(pose.shape) == 3:
        einsum_str = 'ktv,ck->ctv'
    else:
        einsum_str = 'ktvm,ck->ctvm'
    pose_transformed_wo_conf = np.einsum(einsum_str, pose_w_ones, trans_mat)
    pose_transformed = np.concatenate([pose_transformed_wo_conf[:2], conf], axis=0)
    return pose_transformed

class PoseTransform(object):
    """ A general class for applying transformations to pose sequences, empty init returns identity """

    def __init__(self, sx=1, sy=1, tx=0, ty=0, rot=0, shearx=0., sheary=0., flip=False, trans_mat=None):
        """ An explicit matrix overrides all parameters"""
        if trans_mat is not None:
            self.trans_mat = trans_mat
        else:
            self.trans_mat = get_aff_trans_mat(sx, sy, tx, ty, rot, shearx, sheary, flip)

    def __call__(self, x):
        x = apply_pose_transform(x, self.trans_mat)
        return x


trans_list = [
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=False),  # 0
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=True),  # 1
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=False, shearx=0.1, sheary=0.1),  # 2
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=True, shearx=0.1, sheary=0.1),  # 3
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=False, shearx=0, sheary=0.1),  # 4
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=True, shearx=0, sheary=0.1),  # 5
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=False, shearx=0.1, sheary=0),  # 6
    PoseTransform(sx=1, sy=1, tx=0, ty=0, rot=0, flip=True, shearx=0.1, sheary=0),  # 7
]


class PoseSegDataset(Dataset):
    """
    Generates a dataset with two objects, a np array holding sliced pose sequences
    and an object array holding file name, person index and start time for each sliced seq


    If path_to_patches is provided uses pre-extracted patches. If lmdb_file or vid_dir are
    provided extracts patches from them, while hurting performance.
    """

    def __init__(self, path_to_json_dir, path_to_vid_dir=None, normalize_pose_segs=True, return_indices=False,
                 return_metadata=False, debug=False, return_global=True, evaluate=False, abnormal_train_path=None,
                 **dataset_args):
        super().__init__()
        self.args = dataset_args
        self.path_to_json = path_to_json_dir
        self.patches_db = None
        self.use_patches = False
        self.normalize_pose_segs = normalize_pose_segs
        self.headless = dataset_args.get('headless', False)
        self.path_to_vid_dir = path_to_vid_dir
        self.eval = evaluate
        self.debug = debug
        self.sence_id = 0
        num_clips = dataset_args.get('specific_clip', None)
        self.return_indices = return_indices
        self.return_metadata = return_metadata
        self.return_global = return_global
        self.transform_list = dataset_args.get('trans_list', None)
        if self.transform_list is None:
            self.apply_transforms = False
            self.num_transform = 1
        else:
            self.apply_transforms = False
            self.num_transform = 1
            # self.apply_transforms = True
            # self.num_transform = len(self.transform_list)  # self.num_transform:7
        self.train_seg_conf_th = dataset_args.get('train_seg_conf_th', 0.0)
        self.seg_len = dataset_args.get('seg_len', 24)
        self.seg_stride = dataset_args.get('seg_stride', 1)
        # segs_path_data_np
        self.segs_data_np, self.segs_meta, self.person_keys, self.global_data_np, \
        self.global_data, self.segs_score_np,self.sence_id,self.segs_path_data_np = \
            gen_dataset(path_to_json_dir, num_clips=num_clips, ret_keys=True,
                        ret_global_data=return_global, **dataset_args)
        self.segs_meta = np.array(self.segs_meta)
        if abnormal_train_path is not None:
            self.segs_data_np_ab, self.segs_meta_ab, self.person_keys_ab, self.global_data_np_ab, \
            self.global_data_ab, self.segs_score_np_ab,self.sence_id_ab,self.segs_path_data_ab_np = \
                gen_dataset(abnormal_train_path, num_clips=num_clips, ret_keys=True,
                            ret_global_data=return_global, **dataset_args)
            self.segs_meta_ab = np.array(self.segs_meta_ab)
            ab_labels = get_ab_labels(self.segs_data_np_ab, self.segs_meta_ab, path_to_vid_dir, abnormal_train_path)
            # print('ab_labels')
            # print(ab_labels)
            num_normal_samp = self.segs_data_np.shape[0]
            num_abnormal_samp = (ab_labels == -1).sum()
            # if(ab_labels == -1):
            #     print(num_abnormal_samp)
            # print('num_abnormal_samp')
            # print(num_abnormal_samp)
            total_num_normal_samp = num_normal_samp + (ab_labels == 1).sum()
            print(total_num_normal_samp)
            print((ab_labels == 1).sum())
            print((ab_labels == -1).sum())
            print("Num of abnormal sapmles: {}  | Num of normal samples: {}  |  Precent: {}".format(
                num_abnormal_samp, total_num_normal_samp, num_abnormal_samp / total_num_normal_samp))
            # print('num_normal_samp')
            # print(num_normal_samp)
            # self.labels = np.concatenate((np.ones(num_normal_samp), ab_labels),
            #                              axis=0).astype(int)
            # lxy
            self.labels = np.concatenate((np.ones(num_normal_samp), ab_labels),
                                         axis=0).astype(int)
            # print(self.labels)
            # (48671, 3, 24, 18)
            # print(self.segs_data_np.shape)
            self.segs_data_np = np.concatenate((self.segs_data_np, self.segs_data_np_ab), axis=0)
            # (73379, 3, 24, 18)
            # print(self.segs_data_np.shape)
            # print(self.segs_data_np[0])
            self.segs_meta = np.concatenate((self.segs_meta, self.segs_meta_ab), axis=0)
            self.global_data_np = np.concatenate((self.global_data_np, self.global_data_np_ab), axis=0)
            self.segs_score_np = np.concatenate(
                (self.segs_score_np, self.segs_score_np_ab), axis=0)
            self.global_data += self.global_data_ab
            self.sence_id += self.sence_id_ab
            # print(len(self.sence_id))
            self.segs_path_data_np= np.concatenate((self.segs_path_data_np, self.segs_path_data_ab_np), axis=0)
            # print(self.segs_path_data_np.shape)
            self.person_keys.update(self.person_keys_ab)
        else:
            self.labels = np.ones(self.segs_data_np.shape[0])
        # Convert person keys to ints
        self.person_keys = {k: [int(i) for i in v] for k, v in self.person_keys.items()}
        self.metadata = self.segs_meta
        # print('self.segs_data_np.shape')
        # print(self.segs_data_np.shape)
        self.num_samples, self.C, self.T, self.V = self.segs_data_np.shape   # (n,c,t,v) (73379,3,24,18)   (c,t,v)  (3,24,18)

    def __getitem__(self, index):
        # Select sample and augmentation. I.e. given 5 samples and 2 transformations,
        # sample 7 is data sample 7%5=2 and transform is 7//5=1
        # print(index)  # 315453
        # print(self.num_samples)  # 73379
        if self.apply_transforms:   # self.apply_transforms=True
            sample_index = index % self.num_samples
            # print('index')
            # print(index)   # train：0 test:1
            # print('sample_index')
            # print(sample_index)   # train：0 test:1
            trans_index = math.floor(index / self.num_samples)
            # print('trans_index')
            # print(trans_index) # train：0 test:0
            data_numpy = np.array(self.segs_data_np[sample_index])
            # print('data_numpy.shape')
            # print(data_numpy.shape)  # (3,24,18)
            data_transformed = self.transform_list[trans_index](data_numpy)
            # print('data_transformed.shape')
            # print(data_transformed.shape)
        else:
            sample_index = index
            data_transformed = np.array(self.segs_data_np[index])
            trans_index = 0  # No transformations

        # print('self.normalize_pose_segs')
        # print(self.normalize_pose_segs)  # True
        if self.normalize_pose_segs:
            data_transformed = normalize_pose(data_transformed.transpose((1, 2, 0))[None, ...],
                                              **self.args).squeeze(axis=0).transpose(2, 0, 1)

        # print('data_transformed.shape')
        # print(data_transformed.shape)  # (3,24,18)
        ret_arr = [data_transformed, trans_index]

        # ret_arr += [self.segs_score_np[sample_index]]
        # ret_arr += [self.sence_id[sample_index]]
        ret_arr += [self.sence_id[sample_index]]
        ret_arr += [self.labels[sample_index]]
        ret_arr += [self.segs_path_data_np[sample_index]]
        # print(sample_index)
        # print(data_transformed)
        return ret_arr   # [data_transformed, trans_index, self.segs_score_np[sample_index],self.labels[sample_index]]

    def get_all_data(self, normalize_pose_segs=True):
        if normalize_pose_segs:
            segs_data_np = normalize_pose(self.segs_data_np.transpose((0, 2, 3, 1)), **self.args).transpose(
                (0, 3, 1, 2))
        else:
            segs_data_np = self.segs_data_np
        if self.num_transform == 1 or self.eval:
            return list(segs_data_np)
        return segs_data_np

    def __len__(self):
        return self.num_transform * self.num_samples


def get_dataset_and_loader(args, trans_list, only_test=False):
    '''
    batch_size:256
    num_workers:8
    '''
    loader_args = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': False}
    '''
    headless:False
    norm_scale:0  缩放不保持比例
    prop_norm_scale:1 缩放保持比例
    seg_len:24  窗口滑动帧数
    dataset:UBnormal
    train_seg_conf_th：0
    specific_clip：None
    '''
    dataset_args = {'headless': args.headless, 'scale': args.norm_scale, 'scale_proportional': args.prop_norm_scale,
                    'seg_len': args.seg_len, 'return_indices': True, 'return_metadata': True, "dataset": args.dataset,
                    'train_seg_conf_th': args.train_seg_conf_th, 'specific_clip': args.specific_clip}
    dataset, loader = dict(), dict()
    splits = ['train', 'test'] if not only_test else ['test']
    for split in splits:
        evaluate = split == 'test'
        abnormal_train_path = args.pose_path_train_abnormal if split == 'train' else None   # None
        normalize_pose_segs = args.global_pose_segs  # True
        dataset_args['trans_list'] = trans_list[:args.num_transform] if split == 'train' else None  # num_transform：2
        dataset_args['seg_stride'] = args.seg_stride if split == 'train' else 1  # No strides for test set
        dataset_args['vid_path'] = args.vid_path[split]
        dataset[split] = PoseSegDataset(args.pose_path[split], path_to_vid_dir=args.vid_path[split],
                                        normalize_pose_segs=normalize_pose_segs,
                                        evaluate=evaluate,
                                        abnormal_train_path=abnormal_train_path,
                                        **dataset_args)


        loader[split] = DataLoader(dataset[split], **loader_args, shuffle=(split == 'train'))
    if only_test:
        loader['train'] = None
    return dataset, loader


def shanghaitech_hr_skip(shanghaitech_hr, scene_id, clip_id):
    if not shanghaitech_hr:
        return shanghaitech_hr
    if (int(scene_id), int(clip_id)) in SHANGHAITECH_HR_SKIP:
        return True
    return False

'''
这段代码用于生成一个数据集，其中包括以下组件：

segs_data_np：一个 NumPy 数组，用于存储切分的姿势序列。
segs_score_np：一个 NumPy 数组，用于存储切分的分数数据。
segs_meta：一个对象数组，用于存储每个切分序列的文件名、人员索引和开始时间。
global_data：一个列表，用于存储全局数据。
'''
def gen_dataset(person_json_root, num_clips=None, kp18_format=True, ret_keys=False, ret_global_data=True,
                **dataset_args):
    segs_data_np = []
    segs_score_np = []
    segs_meta = []
    global_data = []
    person_keys = dict()
    start_ofst = dataset_args.get('start_ofst', 0)
    seg_stride = dataset_args.get('seg_stride', 1)  # 6
    seg_len = dataset_args.get('seg_len', 24)
    headless = dataset_args.get('headless', False)
    seg_conf_th = dataset_args.get('train_seg_conf_th', 0.0)
    dataset = dataset_args.get('dataset', 'UBnormal')  # UBnormal

    dir_list = os.listdir(person_json_root)
    json_list = sorted([fn for fn in dir_list if fn.endswith('tracked_person.json')])
    scene_dir = []
    if num_clips is not None:
        json_list = [json_list[num_clips]]  # For debugging purposes
    for person_dict_fn in tqdm(json_list):
        if dataset == "UBnormal":
            type, scene_id, clip_id = \
                re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*)_alphapose_.*', person_dict_fn)[0]
            clip_id = type + "_" + clip_id
        else:
            scene_id, clip_id = person_dict_fn.split('_')[:2]
            if shanghaitech_hr_skip(dataset=="ShaghaiTech-HR", scene_id, clip_id):
                continue
        clip_json_path = os.path.join(person_json_root, person_dict_fn)
        with open(clip_json_path, 'r') as f:
            clip_dict = json.load(f)
        clip_segs_data_np, clip_segs_meta, clip_keys, single_pos_np, _, score_segs_data_np = gen_clip_seg_data_np(
            clip_dict, start_ofst,
            seg_stride,
            seg_len,
            scene_id=scene_id,
            clip_id=clip_id,
            ret_keys=ret_keys,
            dataset=dataset)

        # pose_segs_meta.append([int(scene_id), clip_id, int(single_pose_meta[0]), int(start_key)])
        _, _, _, global_data_np, global_data, _ = gen_clip_seg_data_np(clip_dict, start_ofst, 1, 1, scene_id=scene_id,
                                                                       clip_id=clip_id,
                                                                       ret_keys=ret_keys,
                                                                       global_pose_data=global_data,
                                                                       dataset=dataset)
        segs_data_np.append(clip_segs_data_np)
        # print(len(clip_segs_data_np))
        # print(clip_segs_data_np[0].shape) # (24, 17, 3)
        segs_score_np.append(score_segs_data_np)
        scene_tmp = [scene_id]*len(clip_segs_meta)
        # print(clip_segs_meta)
        # scene_dir += scene_tmp
        scene_dir+=clip_segs_meta
        # print(clip_id)
        # print(scene_dir)
        segs_meta += clip_segs_meta
        person_keys = {**person_keys, **clip_keys}

        # print(f'clip_segs_meta:{clip_segs_meta}')
        # print(f'len(clip_segs_meta):{len(clip_segs_meta)}')

    # Global data
    global_data_np = np.expand_dims(np.concatenate(global_data, axis=0), axis=1)
    # print(f'segs_data_np{segs_data_np.}')
    segs_data_np = np.concatenate(segs_data_np, axis=0)
    segs_score_np = np.concatenate(segs_score_np, axis=0)

    # if normalize_pose_segs:
    #     segs_data_np = normalize_pose(segs_data_np, vid_res=vid_res, **dataset_args)
    #     global_data_np = normalize_pose(global_data_np, vid_res=vid_res, **dataset_args)
    #     global_data = [normalize_pose(np.expand_dims(data, axis=0), **dataset_args).squeeze() for data
    #                    in global_data]
    if kp18_format and segs_data_np.shape[-2] == 17:
        # 10,13
        segs_data_np = keypoints17_to_coco18(segs_data_np)
        segs_path_data_np = (segs_data_np[:, :, 10, :2] + segs_data_np[:, :, 13, :2]) / 2
        print(segs_path_data_np.shape)
        segs_path_data_np[:,:,:1] = (segs_path_data_np[:,:,:1])/1080
        segs_path_data_np[:,:,1:2] = (segs_path_data_np[:,:,1:2])/720
        global_data_np = keypoints17_to_coco18(global_data_np)
        global_data = [keypoints17_to_coco18(data) for data in global_data]
    if headless:
        segs_data_np = segs_data_np[:, :, 5:]
        global_data_np = global_data_np[:, :, 5:]
        global_data = [data[:, 5:, :] for data in global_data]

    segs_data_np = np.transpose(segs_data_np, (0, 3, 1, 2)).astype(np.float32)
    global_data_np = np.transpose(global_data_np, (0, 3, 1, 2)).astype(np.float32)

    if seg_conf_th > 0.0:
        segs_data_np, segs_meta, segs_score_np = \
            seg_conf_th_filter(segs_data_np, segs_meta, segs_score_np, seg_conf_th)
    if ret_global_data:
        if ret_keys:
            return segs_data_np, segs_meta, person_keys, global_data_np, global_data, segs_score_np,scene_dir,segs_path_data_np
        else:
            return segs_data_np, segs_meta, global_data_np, global_data, segs_score_np,scene_dir,segs_path_data_np
    if ret_keys:
        return segs_data_np, segs_meta, person_keys, segs_score_np,scene_dir,segs_path_data_np
    else:
        return segs_data_np, segs_meta, segs_score_np,scene_dir,segs_path_data_np


def keypoints17_to_coco18(kps):
    """
    Convert a 17 keypoints coco format skeleton to an 18 keypoint one.
    New keypoint (neck) is the average of the shoulders, and points
    are also reordered.
    """
    kp_np = np.array(kps)
    # print('kp_np')
    # print(kp_np.shape)
    neck_kp_vec = 0.5 * (kp_np[..., 5, :] + kp_np[..., 6, :])
    kp_np = np.concatenate([kp_np, neck_kp_vec[..., None, :]], axis=-2)
    opp_order = [0, 17, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    opp_order = np.array(opp_order, dtype=np.int64)
    kp_coco18 = kp_np[..., opp_order, :]
    return kp_coco18


def seg_conf_th_filter(segs_data_np, segs_meta, segs_score_np, seg_conf_th=2.0):
    # seg_len = segs_data_np.shape[2]
    # conf_vals = segs_data_np[:, 2]
    # sum_confs = conf_vals.sum(axis=(1, 2)) / seg_len
    sum_confs = segs_score_np.mean(axis=1)
    seg_data_filt = segs_data_np[sum_confs > seg_conf_th]
    seg_meta_filt = list(np.array(segs_meta)[sum_confs > seg_conf_th])
    segs_score_np = segs_score_np[sum_confs > seg_conf_th]

    return seg_data_filt, seg_meta_filt, segs_score_np


class UbnormalDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

# # data, trans, scene_id, label
# def gen_fusion_dataset_dataloader():
#     parser = init_parser()
#     args = parser.parse_args()
#     args, model_args = init_sub_args(args)
#     dataset, loader = get_dataset_and_loader(args, trans_list=trans_list)
#     dataset_train = dataset['train']
#     dataset_test = dataset['test']
#     dataset_a = []
#     dataset_n = []
#     dataset_t = []
#     loader_args = {'batch_size': args.batch_size, 'num_workers': 0, 'pin_memory': False}
#     # checkpoints = f'/home/liuxinyu/PycharmProjects/KG-VAD/stage1/checkpoints/trained_model_0.00021018934103932962.pth'
#     # model = Autoencoder(2,1,(3,3),1,True)
#     # model.load_state_dict(torch.load(checkpoints))
#     # adj_matrix = get_adjacency_matrices()
#     # adj_matrix = np.tile(adj_matrix, (3, 1, 1))
#     # adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32).clone().detach()
#     for train in dataset_train:
#         data,tran,mate,label = train
#         # print(f'mate:{mate}')
#         # print(f'mate[1]:{mate[1]}')
#         # print(f'mate[1][0]:{mate[1][0]}')
#         # if(label == 1):
#         print(f'mate:{mate}')
#         counter = 0    # 计数器
#         old_scene = -1
#         old_data = ''
#         for i in range(len(data)):
#             if(old_scene =)
#
#         if(mate[1][0] == 'n'):
#             data = data[:2,:,:]
#             scene = mate[0]
#             scene = torch.load(f"/home/liuxinyu/PycharmProjects/KG-VAD/data/UBnormal_scene_feature/scene{scene}_features.pth")
#             scene = scene.expand(1, 24, 1000)
#             t = [data,mate,scene,label]
#             dataset_n.append(t)
#         elif(mate[1][0] == 'a'):
#             data = data[:2, :, :]
#             scene = mate[0]
#             # print(f'a_mate[0]:{mate[0]}')
#             scene = torch.load(f"/home/liuxinyu/PycharmProjects/KG-VAD/data/UBnormal_scene_feature/scene{scene}_features.pth")
#             # scene = scene.expand(1,1000)
#             scene = scene.expand(1, 24, 1000)
#             t = [data, mate, scene, label]
#             dataset_a.append(t)
#
#     for test in dataset_test:
#         data,tran,mate,label = test
#         data = data[:2, :, :]
#         # data = torch.tensor(data).float()
#         # data = torch.unsqueeze(data, dim=0)
#         # with torch.no_grad():  # Add this line to disable gradient tracking for the following operations
#         #     data = model.encoder(data, adj_matrix)
#         # data = torch.squeeze(data, dim=1)
#         # data = data.cpu().detach().numpy()
#         scene = mate[0]
#         # print(f'test_mate:{mate}')
#         scene = torch.load(f"/home/liuxinyu/PycharmProjects/KG-VAD/data/UBnormal_scene_feature/scene{scene}_features.pth")
#         # scene = scene.expand(1, 1000)
#         scene = scene.expand(1, 24, 1000)
#         t = [data, mate, scene, label]
#         dataset_t.append(t)
#
#     # 使用 CustomDataset 来包装数据列表
#     dataset_n = UbnormalDataset(dataset_n)
#     dataset_a = UbnormalDataset(dataset_a)
#     dataset_t = UbnormalDataset(dataset_t)
#
#     loader_n = DataLoader(dataset_n, **loader_args, shuffle=True,generator=torch.Generator(device='cuda'))
#     loader_a = DataLoader(dataset_a, **loader_args, shuffle=True,generator=torch.Generator(device='cuda'))
#     loader_t = DataLoader(dataset_t, **loader_args, shuffle=False,generator=torch.Generator(device='cuda'))
#
#     return dataset_n,dataset_a,dataset_t,loader_n,loader_a,loader_t

# data, trans, scene_id, label

def uniform_sampling(batch_szie,tmp_train):
    n = len(tmp_train) // batch_szie
    tmp_train_seg = []
    if len(tmp_train) >= batch_szie:
        for i in range(n):
            tmp_train_seg += tmp_train[i::n][:batch_szie]
        return tmp_train_seg
    while len(tmp_train) < batch_szie:
        tmp_train = tmp_train + tmp_train[:batch_szie - len(tmp_train)]
    return tmp_train


def gen_fusion_dataset_dataloader():
    parser = init_parser()
    args = parser.parse_args()
    args, model_args = init_sub_args(args)
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list)
    dataset_train = dataset['train']
    dataset_test = dataset['test']
    dataset_a = []
    dataset_n = []
    dataset_t = []
    loader_args = {'batch_size': args.batch_size, 'num_workers': 0, 'pin_memory': False}
    counter = 0  # 计数器
    old_scene = -1
    old_mate = ''
    old_tran = 0
    tmp_train = []
    nn = 0
    aa = 0
    for train in dataset_train:
        data, tran, mate, label,path_data = train
        data = data[:2, :, :]
        scene = mate[0]
        scene = torch.load(
            f"/home/liuxinyu/PycharmProjects/KG-VAD/data/UBnormal_scene_feature/scene{scene}_features.pth")
        scene = scene.expand(1, 1, 512)
        if(mate[0] == old_scene and mate[1] == old_mate and tran ==old_tran):
            counter+=1
            t = [data, mate, scene, label,path_data]
            tmp_train.append(t)
        else:
            if(counter!=0):
                tmp_train_step = uniform_sampling(args.batch_size,tmp_train)
                # print(f'出现几次？{len(tmp_train_step)}\told_scene:{old_scene}\told_mate:{old_mate}\told_tran:{old_tran}')
                if(old_mate[0] == 'n'):
                    dataset_n+=tmp_train_step
                    nn += 1
                elif(old_mate[0] == 'a'):
                    dataset_a+=tmp_train_step
                    aa += 1
            counter = 1
            old_scene = mate[0]
            old_mate = mate[1]
            old_tran = tran
            tmp_train = []
            t = [data, mate, scene, label,path_data]
            tmp_train.append(t)

    for test in dataset_test:
        data, tran, mate, label,path_data = test
        data = data[:2, :, :]
        scene = mate[0]
        scene = torch.load(
            f"/home/liuxinyu/PycharmProjects/KG-VAD/data/UBnormal_scene_feature/scene{scene}_features.pth")
        # print(f'scene.shape:{scene.shape}')
        scene = scene.expand(1, 1, 512)
        t = [data, mate, scene, label,path_data]
        dataset_t.append(t)

    print(f'aa:{aa}\tnn:{nn}')
    # 使用 CustomDataset 来包装数据列表
    dataset_n = UbnormalDataset(dataset_n)
    dataset_a = UbnormalDataset(dataset_a)
    dataset_t = UbnormalDataset(dataset_t)

    loader_n = DataLoader(dataset_n, **loader_args, shuffle=True, generator=torch.Generator(device='cuda'))
    loader_a = DataLoader(dataset_a, **loader_args, shuffle=True, generator=torch.Generator(device='cuda'))
    loader_t = DataLoader(dataset_t, **loader_args, shuffle=False, generator=torch.Generator(device='cuda'))

    return dataset_n, dataset_a, dataset_t, loader_n, loader_a, loader_t

def gen_fusion_dataset_dataloader_old():
    parser = init_parser()
    args = parser.parse_args()
    args, model_args = init_sub_args(args)
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list)
    dataset_train = dataset['train']
    dataset_test = dataset['test']
    dataset_a = []
    dataset_n = []
    dataset_t = []
    loader_args = {'batch_size': args.batch_size, 'num_workers': 0, 'pin_memory': False}
    counter = 0  # 计数器
    old_scene = -1
    old_mate = ''
    old_tran = 0
    tmp_train = []
    nn = 0
    aa = 0
    for train in dataset_train:
        # data, tran, mate, label = train
        data, tran, mate, label, path_data = train
        data = data[:2, :, :]
        scene = mate[0]
        scene = torch.load(
            f"/home/liuxinyu/PycharmProjects/KG-VAD/data/UBnormal_scene_feature/scene{scene}_features.pth")
        scene = scene.expand(1, 1, 512)
        if(mate[0] == old_scene and mate[1] == old_mate and tran ==old_tran):
            counter+=1
            t = [data, mate, scene, label]
            tmp_train.append(t)
        else:
            if(counter!=0):
                tmp_train_step = uniform_sampling(args.batch_size,tmp_train)
                # print(f'出现几次？{len(tmp_train_step)}\told_scene:{old_scene}\told_mate:{old_mate}\told_tran:{old_tran}')
                if(old_mate[0] == 'n'):
                    dataset_n+=tmp_train_step
                    nn += 1
                elif(old_mate[0] == 'a'):
                    dataset_a+=tmp_train_step
                    aa += 1
            counter = 1
            old_scene = mate[0]
            old_mate = mate[1]
            old_tran = tran
            tmp_train = []
            t = [data, mate, scene, label]
            tmp_train.append(t)

    for test in dataset_test:
        data, tran, mate, label,path_data = test
        data = data[:2, :, :]
        scene = mate[0]
        scene = torch.load(
            f"/home/liuxinyu/PycharmProjects/KG-VAD/data/UBnormal_scene_feature/scene{scene}_features.pth")
        # print(f'scene.shape:{scene.shape}')
        scene = scene.expand(1, 1, 512)
        t = [data, mate, scene, label]
        dataset_t.append(t)

    print(f'aa:{aa}\tnn:{nn}')
    # 使用 CustomDataset 来包装数据列表
    dataset_n = UbnormalDataset(dataset_n)
    dataset_a = UbnormalDataset(dataset_a)
    dataset_t = UbnormalDataset(dataset_t)

    loader_n = DataLoader(dataset_n, **loader_args, shuffle=True, generator=torch.Generator(device='cuda'))
    loader_a = DataLoader(dataset_a, **loader_args, shuffle=True, generator=torch.Generator(device='cuda'))
    loader_t = DataLoader(dataset_t, **loader_args, shuffle=False, generator=torch.Generator(device='cuda'))

    return dataset_n, dataset_a, dataset_t, loader_n, loader_a, loader_t

def get_cluster_dataset():
    parser = init_parser()
    args = parser.parse_args()
    args, model_args = init_sub_args(args)
    dataset, loader = get_dataset_and_loader(args,trans_list=trans_list)
    dataset_train = dataset['train']
    dataset_n = []
    dataset_a = []
    scene_n = []
    scene_a = []
    for train in dataset_train:
        data, tran, mate, label,path = train
        data = data[:2, :, :]
        scene = mate[0]
        if label == 1:
            dataset_n.append(data)
            scene_n.append(scene)
        elif label == -1:
            dataset_a.append(data)
            scene_a.append(scene)
    print(f'正常：{len(dataset_n)}\t异常：{len(dataset_a)}')

    return dataset_n,dataset_a,scene_n,scene_a



from stage1.args import init_parser,init_sub_args

# def main():
#
#     # 创建一个形状为 (3, 24, 18) 的随机数组
#     data_np = np.random.rand(3, 24, 18)
#     data_np = data_np[:2,:,:]
#     print(data_np.shape)

def main():
    parser = init_parser()
    args = parser.parse_args()
    args, model_args = init_sub_args(args)
    dataset, loader = get_dataset_and_loader(args,trans_list=trans_list)
    dataset = dataset['test']
    for i in dataset:
        a,b,c,d = i
        # print(a.shape)
        # print(b)
        # print(c)   # [10, 'abnormal__1', 1, 0]
        print(c[1])
        input_string = c[1]
        parts = input_string.split("__")
        if len(parts) == 2:
            first_part = parts[0]
            second_part = parts[1]
            print("第一部分: ", first_part)
            print("第二部分: ", second_part)
        else:
            print("无法找到分隔符 '__'")
        # print(d)


    # for split, data in dataset.items():
    #     new_data = []
    #     for i in range(len(data)):
    #         print(type(data[i][0]))
    #         # print(data[i][0])
    #         # print(data[i][0][:2, :, :])
    #         new_data = data[i][0][:2, :, :]
    #         data[i][0] = new_data
    #         print(data[i][0].shape)
    #         # print(data[i][0].shape)
    #     dataset[split] = data


        # for split, data in dataset.items():
    #     for i in range(len(data)):
    #         print(data[i][0].shape)






    #     print(f'split{split}')
    # return

# def main():
#     parser = init_parser()
#     args = parser.parse_args()
#     args, model_args = init_sub_args(args)
#     dataset, loader = get_dataset_and_loader(args,trans_list=trans_list)
#     # print(dataset)
#     # print(loader)
#     # print("===========================================================================================")
#     # for split, data_loader in loader.items():
#     #     print(f"Data Split: {split}")
#     #     print(f"Batch Size: {data_loader.batch_size}")
#     #     print(f"Number of Workers: {data_loader.num_workers}")
#     #     print(f"Pin Memory: {data_loader.pin_memory}")
#     #     print("------------")
#     # 可视化 dataset
#     '''
#     dataset
#         - split
#             - train
#             - test
#         - data 有n个下述属性
#             - data_np   (3,24,18)  # ((x坐标、y坐标、置信度),24帧,18个像素点）
#             - trans_index   # 是否进行了数据增强，进行了何种数据增强，默认0。在这好像也是0
#             - seg_score  # 整个视频帧的置信分数
#             - label   # 标签  train:1是正常（4700） -1是异常（142058）   test:1应该是不考虑是否异常（301034）
#
#         data:[
#                 [data_np,trans_index,seg_score,label],
#                 [data_np,trans_index,seg_score,label],
#                 [data_np,trans_index,seg_score,label],
#                 [data_np,trans_index,seg_score,label],
#                 ... ...
#              ]
#     '''
#     a1 = 0
#     a2 = 0
#     a3 = 0
#     b1 = 0
#     b2 = 0
#     b3 = 0
#     for split, data in dataset.items():
#         print(f"{split} dataset:")
#         i = 0
#         data,_,_,_ = data[0]
#         print(data)
#         for sample in data:
#             # sample = data[i]
#             # 这里假设 sample 是一个列表，包含数据和其他信息
#             data_np, trans_index, seg_score, label = sample
#             if split == 'train':
#                 if label == -1:
#                     a1 += 1
#                 elif label == 0:
#                     a2 += 1
#                 elif label == 1:
#                     a3 += 1
#
#             else:
#                 if label == -1:
#                     b1 += 1
#                 elif label == 0:
#                     b2 += 1
#                 elif label == 1:
#                     b3 += 1
#
#     print(a1,a2,a3,b1,b2,b3)

if __name__ == '__main__':
    main()
