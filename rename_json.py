import os

def rename_json_files(folder_path):
    """
    遍历文件夹中的 JSON 文件，并按照规则重命名。

    :param folder_path: 文件夹路径
    :param prefix: 新文件名前缀
    :param start_number: 起始编号
    """
    try:
        # 确保文件夹存在
        if not os.path.isdir(folder_path):
            print(f"文件夹不存在: {folder_path}")
            return

        # 获取文件夹中的所有 JSON 文件
        json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
        if not json_files:
            print("文件夹中没有找到 JSON 文件。")
            return

        # 遍历并重命名文件
        for i, file_name in enumerate(json_files):
            old_path = os.path.join(folder_path, file_name)
            print(old_path)
            new_name = (file_name.split('_')[0])+'_'+(file_name.split('_')[1])+'_'+(file_name.split('_')[2])+'_'+(file_name.split('_')[3])+'_'+str(int(file_name.split('_')[4])+1)+'_alphapose_tracked_person.json'
            print(new_name)
            # new_name = f"{prefix}_{i}.json"
            new_path = os.path.join(folder_path, new_name)

            # 重命名文件
            os.rename(old_path, new_path)
            print(f"已重命名: {file_name} -> {new_name}")

        print("所有文件已成功重命名！")

    except Exception as e:
        print(f"发生错误: {e}")

# 使用示例
folder = "/home/liuxinyu/PycharmProjects/VAD/data/NWPUC/pose/at"  # 替换为你的文件夹路径
rename_json_files(folder)
