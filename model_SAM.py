import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def load_sam_model(model_type='vit_h', checkpoint_path='checkpoints/sam_vit_h_4b8939.pth'):
    """
    加载 SAM 模型。
    :param model_type: SAM 模型类型 ('vit_b', 'vit_l', 'vit_h')，默认为 'vit_h'
    :param checkpoint_path: 模型权重文件路径
    :return: SAM 自动掩码生成器
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # 注册并加载模型
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # 创建自动掩码生成器
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator

def generate_masks(mask_generator, image_path):
    print(f"Processing image: {image_path}")
    
    # 转换为绝对路径（避免中文路径编码问题）
    image_path = os.path.abspath(image_path)
    print(f"Absolute image path: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"File does not exist: {image_path}")
        return None, None

    # 尝试加载图像
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None, None
    
    # 转换颜色格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    return masks, image

def visualize_masks(image, masks, save_path=None):
    """
    在图像上可视化分割掩码。
    :param image: 原始图像 (RGB 格式)
    :param masks: SAM 生成的掩码列表
    :param save_path: 可选，保存可视化结果的路径
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # 绘制每个掩码
    for mask in masks:
        mask_image = mask['segmentation']  # 获取掩码的布尔值区域
        plt.contour(mask_image, colors=[np.random.rand(3,)], linewidths=2)

    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def main():
    # 设置模型类型与权重路径
    model_type = 'vit_h'
    checkpoint_path = 'checkpoints/sam_vit_h_4b8939.pth'

    # 图像路径
    image_path = 'train_folder/显微图像数据02普通辉石_train/显微图像数据02普通辉石/001(-).jpg'

    # 加载模型
    print("Loading SAM model...")
    mask_generator = load_sam_model(model_type, checkpoint_path)

    # 生成掩码
    print(f"Processing image: {image_path}")
    masks, image = generate_masks(mask_generator, image_path)

    # 可视化结果
    print("Visualizing results...")
    visualize_masks(image, masks, save_path='results/visualization.png')

if __name__ == '__main__':
    main()
