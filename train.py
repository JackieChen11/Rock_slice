import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 假设你使用的是ResNet34模型
from model import resnet34

#使用ResNet50模型
from model import resnet50

#使用ResNet101模型
from model import resnet101

#使用SAM模型
from model_SAM import sam_model_registry
# 如果需要额外的函数（如生成和可视化掩码）
from model_SAM import generate_masks
from model_SAM import visualize_masks

def main():
    # 判断设备是否是GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

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
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

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
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val_folder"), transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    print("Using {} images for training, {} images for validation.".format(train_num, val_num))

   # 模型实例化
 
    
    net = resnet101()
    net.to(device)
    # 如果存在已保存的模型，加载它
    if os.path.exists("./resNet101.pth"):
        net.load_state_dict(torch.load("./resNet101.pth", map_location=device))  # map_location 确保设备一致
        print("Loaded the pre-trained model.")
    else:
        print("No pre-trained model found, initializing a new model.")
    """
    #加载初始化模型
    net = resnet101()
    net.to(device)
    """
    # 损失函数与优化器
    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    # 训练参数
    
    epochs = 4
    best_acc = 0.0
    save_path = './resNet101.pth'
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
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss.item())

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
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_accuracy = acc / val_num
        val_accuracies.append(val_accuracy)
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, running_loss / train_steps, val_accuracy))

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
