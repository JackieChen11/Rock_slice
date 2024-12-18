import os
import json

import torch
from PIL import Image
from torchvision import transforms, models
import matplotlib.pyplot as plt

def main():
    # 如果有NVIDA显卡,转到GPU训练，否则用CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 图像预处理
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 加载图片
    img_path = "./val_folder/显微图像数据01橄榄石_val/显微图像数据01橄榄石/005(+).jpg"
    assert os.path.exists(img_path), f"file: '{img_path}' does not exist."
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    # 获取类别映射
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 加载 resnet101 模型
    model = models.resnet101(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 5)  # 调整分类层为 5 个类别
    model = model.to(device)

    # 加载预训练权重
    weights_path = "./resNet101.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' does not exist."
    
    # 加载权重，忽略分类层
    pretrained_weights = torch.load(weights_path, map_location=device)
    model_dict = model.state_dict()
    # 过滤掉分类层的权重
    pretrained_weights = {k: v for k, v in pretrained_weights.items() if k in model_dict and not k.startswith("fc.")}
    model_dict.update(pretrained_weights)
    model.load_state_dict(model_dict)

    # 进入验证阶段
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # 输出预测结果
    print_res = f"class: {class_indict[str(predict_cla)]}   prob: {predict[predict_cla]:.3f}"
    plt.title(print_res)
    for i in range(len(predict)):
        print(f"class: {class_indict[str(i)]:10}   prob: {predict[i].numpy():.3f}")
    plt.show()

      # 收集反馈
    true_class = input("请输入实际类别（如 'class01'）: ").strip()
    feedback_data = {
        "image_path": img_path,
        "predicted_class": f"class{predict_cla:02}",
        "true_class": true_class
    }

    # 保存到反馈文件
    feedback_file = "./feedback_data.json"
    if os.path.exists(feedback_file):
        with open(feedback_file, 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.append(feedback_data)
    with open(feedback_file, 'w') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)

    print("反馈已保存到 feedback_data.json 文件中。")


if __name__ == '__main__':
    main()
