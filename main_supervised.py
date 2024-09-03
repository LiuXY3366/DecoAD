import time

import numpy as np

from stage1.cluster0 import get_cluster
from stage1.cluster2kg import cluster_test
from stage1.knowledge_graph import clean_all, init_anything
from stage1.main_UBnormal import main_UBnormal
from stage2.main_UBnormal2 import main_UBnormal2

if __name__ == '__main__':
    auc_1 = 0
    flag1 = 0
    auc_2 = 0
    flag2 = 0
    initial_epochs_1 = 50
    initial_epochs_2 = 20
    initial_lr1 = 0.0005
    initial_weight_decay1 = 0.00005
    initial_lr2 = 0.000005
    initial_weight_decay2 = 0.000005

    with open('./auc.txt', 'a+') as file:
        for i in range(1):
            epochs_1 = int(initial_epochs_1 / (i+1))
            epochs_2 = initial_epochs_2 - i
            print(f"=====第{i + 1}次训练=====")
            file.write(f"=====第{i + 1}次训练=====\n")

            lr1 = initial_lr1 / (i + 1)
            weight_decay1 = initial_weight_decay1 / (i + 1)
            lr2 = initial_lr2 / (i + 1)
            weight_decay2 = initial_weight_decay2 / (i + 1)

            auc_1, flag1 = main_UBnormal(epochs=epochs_1, auc_2=auc_2, flag2=flag2, lr=lr1, weight_decay=weight_decay1)
            print(f'auc_1:{auc_1}')
            file.write(f'auc_1:{auc_1}\n')
            start_time = time.perf_counter()
            pose_num = get_cluster(auc_1, flag1)
            clean_all()
            init_anything(scene=29, pose=pose_num)
            cluster_test(auc_1, flag1)
            # cost_time = time.perf_counter() - start_time
            # print("========================================================")
            # print(np.sum(cost_time))  # 3691.3083049989655
            # print("FPS is: ", 92640 / (np.sum(cost_time) + 3691.3083049989655))
            # print("========================================================")

            auc_2, flag2 = main_UBnormal2(epochs=epochs_2, auc_1=auc_1, flag1=flag1, lr=lr2, weight_decay=weight_decay2)
            print(f'auc_2:{auc_2}')

