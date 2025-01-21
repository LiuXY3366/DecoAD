import time

from UNVAD.KG.cluster0 import get_cluster, get_scene_cluster
# from UNVAD.KG.cluster import get_cluster
from UNVAD.KG.cluster2kg import cluster_test, cluster_all_test
from UNVAD.KG.knowledge_graph import clean_all, init_anything
from UNVAD.stage1.dataset import gen_fusion_dataset_dataloader
from UNVAD.stage2.dataset2 import gen_fusion_dataset_dataloader_2
from stage1.main import main
from stage2.main2 import main2
from UNVAD.stage1.args import init_parser

# 初始化解析器

parser = init_parser()

# 解析参数
args = parser.parse_args()

if __name__ == '__main__':

    auc_1 =  0.68815
    threshold1 = 36808.23008
    # threshold1 = 112188.04453
    flag1 = 1
    threshold2 = threshold1
    auc_2 = auc_1
    flag2 = flag1

    init_max_auc = auc_2

    initial_epochs_1 = 3000
    initial_epochs_2 = 15
    '''
    initial_lr1 = 0.00001
    initial_weight_decay1 = 0.000005
    '''
    initial_lr1 = 5e-1  # 0.001
    initial_weight_decay1 = 1e-3  #  0.00005
    initial_lr2 = 1e-5
    initial_weight_decay2 = 5e-7

    _, _, train_nloader, test_loader = gen_fusion_dataset_dataloader()
    dataset, dataset_a, dataset_t, loaders, loader_a, loader_t = None,None,None,None,None,None
    for i in range(100):
        epochs_1 = initial_epochs_1
        epochs_2 = initial_epochs_2
        print(f"=====第{i + 1}次训练=====")
        with open('./auc.txt', 'a+') as file:
            file.write(f"=====第{i + 1}次训练=====\n")
        if i % 2 == 1:
            initial_lr1 = initial_lr1 * 0.5
            initial_weight_decay1 = initial_weight_decay1 * 0.5
            initial_lr2 = initial_lr2 * 0.5
            initial_weight_decay2 = initial_weight_decay2 * 0.5
        elif i % 2 == 0 and i != 0:
            initial_lr1 = initial_lr1 * 0.2
            initial_weight_decay1 = initial_weight_decay1 * 0.2
            initial_lr2 = initial_lr2 * 0.2
            initial_weight_decay2 = initial_weight_decay2 * 0.2
        auc_1, flag1, threshold1, train_nloader, test_loader = main(epochs=epochs_1, auc_2=auc_2, flag2=flag2, lr=initial_lr1, weight_decay=initial_weight_decay1,threshold = threshold2,train_nloader=train_nloader,test_loader=test_loader)
        print(f'auc_1:{auc_1}')
        with open('./auc.txt', 'a+') as file:
            file.write(f'auc_1:{auc_1}\tinitial_lr1:{initial_lr1}\tinitial_weight_decay1:{initial_weight_decay1}\n')

        if auc_1 != init_max_auc or i == 0:
            init_max_auc = auc_1
            pose_num, dataset_input, loader_input = get_cluster(auc_1, flag1, threshold1)
            if args.dataset == 'UFSR':
                scene_num = get_scene_cluster(40)
                clean_all()
                init_anything(scene=40, pose=pose_num)
                cluster_all_test(auc_1, flag1, threshold1, dataset_input, loader_input)
            else:
                clean_all()
                init_anything(scene=400, pose=pose_num)
                cluster_test(auc_1, flag1, threshold1, dataset_input, loader_input)

            dataset, dataset_a, dataset_t, loaders, loader_a, loader_t = gen_fusion_dataset_dataloader_2(
                auc_1=auc_1,
                flag1=flag1,
                threshold1 = threshold1)

        auc_2, flag2,threshold2 = main2(epochs=epochs_2, auc_1=auc_1, flag1=flag1, lr=initial_lr2, weight_decay=initial_weight_decay2,threshold = threshold1,dataset=dataset, dataset_a=dataset_a, dataset_t=dataset_t, loaders=loaders, loader_a=loader_a, loader_t=loader_t)
        print(f'auc_2:{auc_2}')
        with open('./auc.txt', 'a+') as file:
            file.write(f'auc_2:{auc_2}\tinitial_lr2:{initial_lr2}\tinitial_weight_decay2:{initial_weight_decay2}\n')