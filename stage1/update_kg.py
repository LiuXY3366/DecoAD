import os
import shutil
import torch
from knowledge_graph import search_all_relation, create_relation
import re
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
from stage1.cluster import get_cluster
from stage1.knowledge_graph import clean_all, init_anything, clean_no_relation

def cal_sim(pose1,pose2,scene1,scene2):
    # 将张量从 GPU 移动到 CPU，并转换为 NumPy 数组
    if pose1.is_cuda:
        pose1 = pose1.cpu().numpy()
    else:
        pose1 = pose1.numpy()

    if pose2.is_cuda:
        pose2 = pose2.cpu().numpy()
    else:
        pose2 = pose2.numpy()

    if scene1.is_cuda:
        scene1 = scene1.cpu().numpy()
    else:
        scene1 = scene1.numpy()

    if scene2.is_cuda:
        scene2 = scene2.cpu().numpy()
    else:
        scene2 = scene2.numpy()
    pose_sim = 1 - distance.cosine(pose1, pose2)
    scene_sim = 1 - distance.cosine(scene1, scene2)
    return pose_sim,scene_sim,(pose_sim+scene_sim)/2

def getNodeAndRela(file_name = '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/main_relation.txt'):
    rps = search_all_relation()
    if os.path.exists(file_name):
        os.remove(file_name)
    for r in rps:
        # print(rps[r])

        # 示例字符串
        text = "(scene15)-[:normal {}]->(pose1)"
        text = str(rps[r])

        # 正则表达式模式
        pattern = r"\((.*?)\)-\[:(.*?) \{\}\]->\((.*?)\)"

        # 使用re模块的search函数查找匹配的字符串
        match = re.search(pattern, text)

        if match:
            scene = match.group(1)
            relation = match.group(2)
            pose = match.group(3)
            # print(f"Scene: {scene}")
            # print(f"Relation: {relation}")
            # print(f"Pose: {pose}")
            print(f"{relation}\t{scene}\t{pose}")
            # 打开文件进行写入
            with open(file_name, "a") as file:
                file.write(f"{relation}\t{scene}\t{pose}\n")

        else:
            print("No match found")

def make_scene(dataset,file_name = '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/main_scenes.txt'):
    if os.path.exists(file_name):
        os.remove(file_name)
    # 这里还需要修改一下
    for i in range(13):
        # 加载 .pth 文件
        feature_path = f"/home/liuxinyu/PycharmProjects/KG-VAD/data/{dataset}_scene_feature/scene{i+1}_features.pth"
        scenes_features = torch.load(feature_path)

        # 将张量转换为列表
        tensor_list = scenes_features.tolist()
        # 打开文件进行写入
        with open(file_name, "a") as file:
            for row in tensor_list:
                # 将每个元素转换为字符串并用空格连接
                row_str = ' '.join(map(str, row))
                file.write(f'scene{i+1} '+row_str + '\n')

def make_pose_dict(filename = '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/main_poses.txt'):
    # 初始化一个空字典
    data_dict = {}

    # 打开并读取文件
    with open(filename, "r") as file:
        for line in file:
            # 去除行尾的换行符，并分割字符串
            parts = line.strip().split()

            # 第一个部分是键
            key = parts[0]

            # 剩余部分是数值，转换为浮点型数组
            values = np.array([float(x) for x in parts[1:]])

            # 将数值数组转换为张量
            tensor_values = torch.tensor(values)

            # 将键值对加入字典
            data_dict[key] = tensor_values
    # # 打印结果
    # for key, value in data_dict.items():
    #     print(f"{key}: {value}")

    return data_dict

def make_scene_dict(filename = '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/main_scenes.txt'):

    # 初始化一个空字典
    data_dict = {}

    # 打开并读取文件
    with open(filename, "r") as file:
        for line in file:
            # 去除行尾的换行符，并分割字符串
            parts = line.strip().split()

            # 第一个部分是键
            key = parts[0]

            # 剩余部分是数值，转换为浮点型数组
            values = np.array([float(x) for x in parts[1:]])

            # 将数值数组转换为张量
            tensor_values = torch.tensor(values)

            # 将键值对加入字典
            data_dict[key] = tensor_values

    # # 打印结果
    # for key, value in data_dict.items():
    #     print(f"{key}: {value}")
    return data_dict

def make_relation_arrs(filename = '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/main_relation.txt'):
    # 初始化三个空数组
    relation_arr = []
    scene_arr = []
    pose_arr = []

    # 打开并读取文件
    with open(filename, "r") as file:
        for line in file:
            # 去除行尾的换行符，并分割字符串
            parts = line.strip().split()

            # 检查是否有三个部分
            if len(parts) == 3:
                # 将每个部分加入相应的数组
                relation_arr.append(parts[0])
                scene_arr.append(parts[1])
                if parts[0][0] == 'n':
                    pose_arr.append('N'+parts[2])
                elif parts[0][0] == 'a':
                    pose_arr.append('A'+parts[2])

    # # 打印结果
    # print("Array 1:", relation_arr)
    # print("Array 2:", scene_arr)
    # print("Array 3:", pose_arr)
    return relation_arr,scene_arr,pose_arr

def count_non_empty_lines(file_name):
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
            non_empty_lines = [line for line in lines if line.strip() != '']
        return len(non_empty_lines)
    except Exception as e:
        print(f"Error occurred: {e}")
        return 0

# 构建子图
# construct == True开始构建子图
def make_sub_kg(construct = False,auc_1 = 0.78500,flag1 = 1):
    if construct:
        clean_all()
        init_anything(scene=400, pose=600)
        get_cluster(auc_1,flag1)
        clean_no_relation()
        print('sub图构建完成！')
        source_file = '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/cluster_centers.txt'
        destination_file = '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/sub_poses.txt'
        shutil.copy(source_file, destination_file)
        getNodeAndRela('/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/sub_relation.txt')
        make_scene('ShanghaiTech','/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/sub_scenes.txt')

# add == True:向所有main_txt添加新的数据
def add_data(add = False):
    main_poses_txt = '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/main_poses.txt'
    sub_poses_txt = '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/sub_poses.txt'
    main_scenes_txt = '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/main_scenes.txt'
    sub_scenes_txt = '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/sub_scenes.txt'
    main_relation_txt = '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/main_relation.txt'
    sub_relation_txt = '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/sub_relation.txt'
    step_relation_file = '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/step_relation.txt'
    main_pose_dict = make_pose_dict(main_poses_txt)
    sub_pose_dict = make_pose_dict(sub_poses_txt)
    main_scene_dict = make_scene_dict(main_scenes_txt)
    sub_scene_dict = make_scene_dict(sub_scenes_txt)
    main_relation_arr,main_scene_arr,main_pose_arr = make_relation_arrs(main_relation_txt)
    sub_relation_arr,sub_scene_arr,sub_pose_arr = make_relation_arrs(sub_relation_txt)

    pose_num = count_non_empty_lines(main_poses_txt)
    scene_num = count_non_empty_lines(main_scenes_txt)
    # print(pose_num)
    # print(scene_num)

    sub_scene_step_arr = []
    sub_pose_step_arr = []
    sub_pose_name_dict = {}
    sub_scene_name_dict = {}

    if os.path.exists(step_relation_file):
        os.remove(step_relation_file)

    with tqdm(total=len(main_relation_arr)*len(sub_relation_arr)) as pbar:
        for i in range(len(sub_relation_arr)):
            sub_relation = sub_relation_arr[i]
            sub_scene_id = sub_scene_arr[i]
            sub_pose_id = sub_pose_arr[i]
            sub_pose_feature = sub_pose_dict[sub_pose_id]
            sub_scene_feature = sub_scene_dict[sub_scene_id]
            max_cos_sim = 0
            max_relation = ''
            for j in range(len(main_relation_arr)):
                main_relation = main_relation_arr[j]
                main_scene_id = main_scene_arr[j]
                main_pose_id = main_pose_arr[j]
                main_pose_feature = main_pose_dict[main_pose_id]
                main_scene_feature = main_scene_dict[main_scene_id]
                pose_sim,scene_sim,cos_sim = cal_sim(sub_pose_feature,main_pose_feature,sub_scene_feature,main_scene_feature)
                if cos_sim > max_cos_sim:
                    max_relation = main_relation
                    max_cos_sim = cos_sim
                pbar.update(1)
            if max_relation == sub_relation:
                # print(f'sub_scene_id:{sub_scene_id}\tsub_pose_id:{sub_pose_id}\tmain_scene_id:{main_scene_id}\tmain_pose_id:{main_pose_id}\trelation:{sub_relation}\tcos_sim:{cos_sim}\tpose_sim:{pose_sim}\tscene_sim:{scene_sim}')
                if pose_sim>=0.7:
                    sub_pose_id = main_pose_id
                    sub_pose_feature = main_pose_feature
                else:
                    sub_pose_id = sub_pose_id+'_sub'
                if scene_sim>=0.7:
                    sub_scene_id = main_scene_id
                    sub_scene_feature = main_scene_feature
                else:
                    sub_scene_id = sub_scene_id+'_sub'
                if pose_sim>=0.2 and scene_sim>=0.6:
                    # print(f'{sub_relation} {sub_scene_id} {sub_pose_id}\n')
                    # 打开文件（如果文件不存在会自动创建）
                    if pose_sim<0.7 or scene_sim<0.7:
                        with open(step_relation_file, 'a') as file:
                            # 写入内容
                            if add:
                                file.write(f'{sub_relation} {sub_scene_id} {sub_pose_id}\n')
                            if 'sub' in sub_scene_id:
                                if sub_scene_id not in sub_scene_step_arr:
                                    sub_scene_step_arr.append(sub_scene_id)
                                    sub_scene_id_new = 'scene'+str(len(sub_scene_step_arr)+scene_num)
                                    sub_scene_name_dict[sub_scene_id] = sub_scene_id_new

                                    tensor_list = sub_scene_feature.tolist()
                                    # print(tensor_list)
                                    with open(main_scenes_txt, 'a') as file:
                                        # 写入内容
                                        row_str = ' '.join(map(str, tensor_list))
                                        # print(row_str)
                                        if add:
                                            file.write(f'{sub_scene_id_new} ' + row_str + '\n')
                                else:
                                    sub_scene_id_new = sub_scene_name_dict[sub_scene_id]
                            else:
                                sub_scene_id_new = sub_scene_id

                            if 'sub' in sub_pose_id:
                                if sub_pose_id not in sub_pose_step_arr:
                                    sub_pose_step_arr.append(sub_pose_id)
                                    if sub_relation == 'abnormal':
                                        sub_pose_id_new = 'Apose'+str(len(sub_pose_step_arr)+pose_num)
                                    else:
                                        sub_pose_id_new = 'Npose' + str(len(sub_pose_step_arr) + pose_num)
                                    sub_pose_name_dict[sub_pose_id] = sub_pose_id_new
                                    tensor_list = sub_pose_feature.tolist()
                                    # print(tensor_list)
                                    with open(main_poses_txt, 'a') as file:
                                        # 写入内容
                                        row_str = ' '.join(map(str, tensor_list))
                                        # print(row_str)
                                        if add:
                                            file.write(f'{sub_pose_id_new} ' + row_str + '\n')
                                else:
                                    sub_pose_id_new = sub_pose_name_dict[sub_pose_id]
                            else:
                                sub_pose_id_new = sub_pose_id
                            with open(main_relation_txt, 'a') as file:
                                # 写入内容
                                print(f'{sub_relation} {sub_scene_id_new} {sub_pose_id_new[1:]}\n')
                                if add:
                                    file.write(f'{sub_relation} {sub_scene_id_new} {sub_pose_id_new[1:]}\n')

# 重新构建知识图谱  update == True
def update_main_kg(update = False):
    file_path = '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/main_relation.txt'
    if update:
        clean_all()
        init_anything(scene=400, pose=600)
        with open(file_path, 'r') as file:
            for line in file:
                # 去掉行末的换行符并拆分字符串
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    relation, scene, pose = parts
                    print(scene, pose, relation)
                    create_relation('scene', 'pose', scene, pose, relation)
        clean_no_relation()
        print('main图更新完成！')






if __name__ == '__main__':
    # getNodeAndRela()
    # make_scene()
    # =========================================
    # getNodeAndRela()
    # pose_dict = make_pose_dict()
    # scene_dict = make_scene_dict()
    # relation_arr,scene_arr,pose_arr = make_relation_arrs()
    # # =========================================
    # make_sub_kg()
    # add_data()
    # update_main_kg(False)
    main_scenes_txt = '/home/liuxinyu/PycharmProjects/KG-VAD/stage1/data/cluster/main_scenes.txt'
    scene_num = count_non_empty_lines(main_scenes_txt)