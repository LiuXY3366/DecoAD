import os
import json
from posixpath import basename
import argparse
import shutil


def convert_data_format(data, split='None'):
    if split == 'testing':
        num_digits = 3
    elif split == 'training':
        num_digits = 4
    elif split == 'None':
        num_digits = 4

    data_new = dict()
    for item in data:
        frame_idx_str = item['image_id'][:-4]  # '0.jpg' -> '0'
        frame_idx_str = frame_idx_str.zfill(num_digits)  # '0' -> '000'
        person_idx_str = str(item['idx'])
        keypoints = item['keypoints']
        scores = item['score']
        if not person_idx_str in data_new:
            data_new[person_idx_str] = {frame_idx_str: {'keypoints': keypoints, 'scores': scores}}
        else:
            data_new[person_idx_str][frame_idx_str] = {'keypoints': keypoints, 'scores': scores}

    return data_new


def read_convert_write(in_full_fname, out_full_fname):
    # Read results file
    with open(in_full_fname, 'r') as f:
        data = json.load(f)

    # Convert reults file format
    data_new = convert_data_format(data)

    # 3. Write
    save = True  # False
    if save:
        with open(out_full_fname, 'w') as f:
            json.dump(data_new, f)


def create_command(alphapose_dir, video_filename, out_dir, is_video=False):
    command_args = {'cfg': os.path.join(alphapose_dir, 'configs/coco/resnet/256x192_res152_lr1e-3_1x-duc.yaml'),
                    'checkpoint': os.path.join(alphapose_dir, 'pretrained_models/fast_421_res152_256x192.pth'),
                    'outdir': out_dir}

    # command = "python ./AlphaPose/scripts/demo_inference.py"
    command = "python scripts/demo_inference.py"

    # loop over command line argument and add to command
    for key, val in command_args.items():
        command += ' --' + key + ' ' + val
    if is_video:
        command += ' --video ' + video_filename
    else:
        command += ' --indir ' + video_filename
    command += ' --sp'  # Torch Re-ID Track
    command += ' --pose_track'  # Torch Re-ID Track
    # command += ' --detector yolox-x'  # Simple MOT Track

    return command

#  python gen_data.py --alphapose_dir /path/to/AlphaPoseFloder/ --dir /input/dir/ --outdir /output/dir/ [--video]
def main():
    # parse command line
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default = '/home/liuxinyu/Desktop/AvenueDataset/testing_videos',dest='dir', type=str, required=False, help='video\images dir')
    ap.add_argument('--outdir',default = '/home/liuxinyu/PycharmProjects/KG-VAD-STC/data/HR-Avenue/pose/test', dest='outdir', type=str, required=False, help='video\images outdir')
    ap.add_argument('--alphapose_dir',default = '/home/liuxinyu/PycharmProjects/KG-VAD/utils/AlphaPose', dest='alphapose_dir', type=str, required=False, help='alphapose_dir')
    ap.add_argument('--video', default=True,dest='video', action='store_true', help='is video')

    # args = vars(ap.parse_args())
    args = ap.parse_args()
    # print(args)
    root = args.dir
    # 获取当前正在运行的 Python 文件所在的目录的绝对路径
    curr_dir = os.path.dirname(os.path.realpath(__file__))  # /home/liuxinyu/桌面/KG-VAD/stage1
    img_dirs = []
    # 获取指定目录中的所有文件和子目录的列表，并将其存储在 output_files 变量中
    output_files = os.listdir(args.outdir)
    # os.walk(root):遍历指定目录 root 及其子目录，返回每个目录的路径、子目录列表和文件列表。
    for path, subdirs, files in os.walk(root):
        # 遍历每个文件的名称
        for name in files:
            run_pose = False
            print(run_pose)
            # if args.video and name.endswith(".mp4") or name.endswith("avi"):
            if args.video and name.endswith(".mp4") or name.endswith("avi"):
                # 处理视频文件时，这一行将构建完整的视频文件路径，并将其存储在 video_filename 变量中
                video_filename = os.path.join(path, name)
                # 从 video_filename 中提取文件的基本名称（不包括路径和扩展名），并将其存储在 video_basename 变量中
                video_basename = basename(video_filename)[:-4]
                # 需要执行姿势估计操作
                run_pose = True
            elif name.endswith(".jpg") or name.endswith(".png"):
                # 如果文件是图像文件，并且文件所在的目录路径 path 不在 img_dirs 列表中，这一行会进一步检查。
                if path not in img_dirs:
                    video_filename = path
                    img_dirs.append(path)
                    video_basename = basename(video_filename)
                    run_pose = True
            if run_pose:
                # Rename constants
                alphapose_orig_results_filename = 'alphapose-results.json'
                alphapose_tracked_results_filename = video_basename + '_alphapose_tracked_person.json'
                alphapose_results_filename = video_basename + '_alphapose-results.json'
                print(alphapose_results_filename)
                if alphapose_results_filename in output_files:
                    continue
                # Change to AlphaPose dir
                # 将当前工作目录更改为 AlphaPose 的工作目录（args.alphapose_dir），以便后续的命令执行在该目录下进行
                os.chdir(args.alphapose_dir)

                # Build command line
                command = create_command(args.alphapose_dir, video_filename, args.outdir, is_video=args.video)
                # Run command
                print('\n$', command)
                os.system(command)

                # Change back to directory containing this script (main_alpahpose.py)
                os.chdir(args.outdir)

                # Convert alphapose-results.json to *_alphapose_tracked_person.json
                read_convert_write(alphapose_orig_results_filename, alphapose_tracked_results_filename)
                # Optionally, rename generic filename 'alphapose-results.json' by adding video filename prefix
                os.rename("alphapose-results.json", alphapose_results_filename)
                shutil.rmtree('poseflow', ignore_errors=True)
                os.chdir(curr_dir)


if __name__ == '__main__':
    main()
