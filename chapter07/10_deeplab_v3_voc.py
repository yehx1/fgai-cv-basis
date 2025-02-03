import os
import torch
import torchvision.models.segmentation as models
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# 自定义 VOC 数据集加载类
class CustomVOCSegmentation(VOCSegmentation):
    def __init__(self, root, year, image_set, transform=None, target_transform=None):
        self.root = root
        self.image_set = image_set
        super().__init__(root=root, year=year, image_set=image_set, transform=transform, target_transform=target_transform, download=False)

# =====================
# 定义训练函数
# =====================
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(train_loader):
        images = images.to(device)
        targets = (targets.squeeze(1) * 255).long().to(device)
        targets[targets == 255] = 0
        # 前向传播
        outputs = model(images)['out']
        loss = criterion(outputs, targets)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

# =====================
# 定义验证函数
# =====================
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = images.to(device)
            targets = targets.squeeze(1).long().to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    return val_loss / len(val_loader)

# =====================
# 可视化验证结果
# =====================
def visualize(model, val_loader, save_path, device):
    model.eval()
    images, targets = next(iter(val_loader))
    images = images.to(device)
    targets = (targets.squeeze(1) * 255).long()
    targets[targets == 255] = 0
    with torch.no_grad():
        outputs = model(images)['out']
    predictions = torch.argmax(outputs, dim=1).cpu()
    for i in range(1):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(images[i].cpu().permute(1, 2, 0).numpy())
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(targets[i].cpu().numpy(), cmap='jet')
        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(predictions[i].cpu().numpy(), cmap='jet')
        plt.savefig(save_path)

if __name__ == "__main__":
    # 创建结果保存路径
    current_file_path = Path(__file__).resolve()
    current_directory = current_file_path.parent
    os.makedirs(f"{current_directory}/result", exist_ok=True)
    # 模型保存路径
    model_path = f"{current_directory}/result/deeplabv3_voc2007_epoch_final.pth"

    # =====================
    # 1. 数据准备
    # =====================
    data_dir = f"{current_directory}/../01_data/VOC"
    image_transform = transforms.Compose([
        transforms.Resize((513, 513)),
        transforms.ToTensor()
    ])
    target_transform = transforms.Compose([
        transforms.Resize((513, 513), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    train_dataset = CustomVOCSegmentation(root=data_dir, year='2007', image_set='train', transform=image_transform, target_transform=target_transform)
    val_dataset = CustomVOCSegmentation(root=data_dir, year='2007', image_set='train', transform=image_transform, target_transform=target_transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # =====================
    # 2. 定义模型
    # =====================
    model = models.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 21, kernel_size=1)  # VOC 有 21 类
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if os.path.exists(model_path):
        print(f"加载已有模型: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("未找到已有模型，从头开始训练。")

    # =====================
    # 3. 定义损失函数和优化器
    # =====================
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # =====================
    # 4. 主训练循环
    # =====================
    num_epochs = 1000
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), model_path)
    torch.save(model.state_dict(), model_path)
    visualize(model, val_loader, f"{current_directory}/result/deeplabv3_infer_sample.jpg", device)
