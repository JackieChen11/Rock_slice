import os
import shutil

def move_folders(src_folder, train_folder, val_folder):
    # 如果目标文件夹不存在，则创建它们
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    # 遍历源文件夹中的文件夹
    for foldername in os.listdir(src_folder):
        folder_path = os.path.join(src_folder, foldername)

        # 只处理文件夹
        if os.path.isdir(folder_path):
            print(f"处理文件夹: {foldername}")  # 打印正在处理的文件夹

            # 以'_train'结尾的文件夹移到训练集文件夹
            if '_train' in foldername.lower():
                shutil.move(folder_path, os.path.join(train_folder, foldername))
            # 以'_val'结尾的文件夹移到验证集文件夹
            elif '_val' in foldername.lower():
                shutil.move(folder_path, os.path.join(val_folder, foldername))
            else:
                print(f"未找到符合条件的文件夹: {foldername}")
        else:
            print(f"忽略非文件夹项: {foldername}")

# 使用示例
src_folder = r'D:/vscode space/岩石薄片'  # 源文件夹路径
train_folder = r'D:/vscode space/岩石薄片/train_folder'  # 训练集文件夹路径
val_folder = r'D:/vscode space/岩石薄片/val_folder'  # 验证集文件夹路径

move_folders(src_folder, train_folder, val_folder)
