import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from model_SAM import sam_model_registry  # SAM模型加载函数
from model import resnet101  # 使用ResNet101模型
import json,sys

def main():
    # 判断设备是否是GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    # 数据预处理方法
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # 数据路径
    data_root = os.path.abspath(os.getcwd())
    image_path = os.path.join(data_root, "")
    assert os.path.exists(image_path), f"{image_path} path does not exist."

    # 加载训练集与验证集
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train_folder"), transform=data_transform["train"])
    train_num = len(train_dataset)

    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())

    # 保存类别字典
    with open('class_indices.json', 'w') as json_file:
        json.dump(cla_dict, json_file, indent=4)

    # 加载数据
    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print(f'Using {nw} dataloader workers every process')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val_folder"), transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    print(f"Using {train_num} images for training, {val_num} images for validation.")

    # 加载预训练模型（ResNet101或SAM模型）
    model_choice = "SAM"  # 可以选择 'resnet101' 或 'SAM'
    
    if model_choice == "resnet101":
        net = resnet101()
        net.to(device)
        # 如果存在已保存的模型，加载它
        if os.path.exists("./resNet101.pth"):
            net.load_state_dict(torch.load("./resNet101.pth", map_location=device))
            print("Loaded the pre-trained ResNet101 model.")
        else:
            print("No pre-trained model found, initializing a new model.")
    
    elif model_choice == "SAM":
        # 加载SAM模型
        sam_model = sam_model_registry("SAM")  # 这假设SAM模型有这个接口
        sam_model.to(device)

        # 定义分类头
        class RockSliceClassifier(nn.Module):
            def __init__(self, sam_model, num_classes):
                super(RockSliceClassifier, self).__init__()
                self.sam_model = sam_model
                self.fc = nn.Linear(256, num_classes)  # 假设SAM模型提取的特征维度是256

            def forward(self, x):
                features = self.sam_model(x)
                features = features.flatten(1)  # 扁平化为一维向量
                out = self.fc(features)  # 分类头
                return out

        net = RockSliceClassifier(sam_model, num_classes=len(flower_list))
        net.to(device)
    
    # 损失函数与优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    # 训练参数
    epochs = 3
    best_acc = 0.0
    save_path = './SAM_model.pth'
    train_steps = len(train_loader)

    # 用于记录训练与验证的准确率
    train_losses = []
    val_accuracies = []

    # 训练过程
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = f"train epoch[{epoch+1}/{epochs}] loss:{loss.item():.3f}"

        # 记录训练损失
        train_losses.append(running_loss / train_steps)

        # 验证过程
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                val_bar.desc = f"valid epoch[{epoch+1}/{epochs}]"

        val_accuracy = acc / val_num
        val_accuracies.append(val_accuracy)
        print(f'[epoch {epoch+1}] train_loss: {running_loss/train_steps:.3f}  val_accuracy: {val_accuracy:.3f}')

        # 保存最好的模型
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)

    print('Finished Training')

    # 训练和验证结果图表
    plt.figure()
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.plot(range(epochs), val_accuracies, label="Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.title('Training and Validation Progress')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
