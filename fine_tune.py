import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from model import resnet34

# 定义反馈数据集类
class FeedbackDataset(Dataset):
    def __init__(self, feedback_file, data_transform):
        with open(feedback_file, 'r') as f:
            self.feedback_data = json.load(f)
        self.data_transform = data_transform

    def __len__(self):
        return len(self.feedback_data)

    def __getitem__(self, idx):
        data = self.feedback_data[idx]
        img_path = data['image_path']
        true_class = int(data['true_class'].replace('class', ''))  # 转换为整数类别
        img = Image.open(img_path)
        if self.data_transform:
            img = self.data_transform(img)
        return img, true_class

def main():
    # 数据变换
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载反馈数据集
    feedback_file = './feedback_data.json'
    dataset = FeedbackDataset(feedback_file, data_transform)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 加载预训练模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet34(num_classes=5).to(device)

    # 载入原模型权重
    weights_path = "./resNet101.pth"
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # 冻结特征提取层，只更新分类层
    for param in model.parameters():
        param.requires_grad = False
    model.fc.requires_grad = True  # 只解冻全连接层

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # 微调模型
    epochs = 5
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(data_loader):.4f}")

    # 保存微调后的模型
    fine_tuned_weights_path = "./resNet101_finetuned.pth"
    torch.save(model.state_dict(), fine_tuned_weights_path)
    print(f"微调后的模型已保存到 {fine_tuned_weights_path}")

if __name__ == '__main__':
    main()
