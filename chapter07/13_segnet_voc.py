import os
import torch
from torch import nn
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
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
# 定义 SegNet 模型
# =====================
class SegNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=21):
        super(SegNet, self).__init__()
        # 编码器
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
            )
        ])

        # 解码器
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.MaxUnpool2d(kernel_size=2, stride=2),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.MaxUnpool2d(kernel_size=2, stride=2),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.MaxUnpool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.MaxUnpool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.MaxUnpool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, kernel_size=3, padding=1)
            )
        ])

    def forward(self, x): # torch.Size([b, 3, 256, 256])
        indices = []
        # [torch.Size([b, 64, 256, 256]), torch.Size([b, 128, 128, 128]), torch.Size([b, 256, 64, 64]), torch.Size([b, 512, 32, 32]), torch.Size([b, 512, 16, 16])]
        sizes = []
        # encode block1: torch.Size([b, 3, 128, 128])
        # encode block2: torch.Size([b, 256, 64, 64])
        # encode block3: torch.Size([b, 512, 32, 32])
        # encode block4: torch.Size([b, 512, 16, 16])
        # encode block5: torch.Size([b, 512, 4, 4])
        for enc_block in self.encoder:
            x = enc_block[:-1](x)      # 先执行卷积和激活操作
            sizes.append(x.size())     # 记录输入到池化前的尺寸
            x, ind = enc_block[-1](x)  # 执行池化操作并获取池化索引
            indices.append(ind)        # 保存索引
        # decode block5: torch.Size([b, 512, 16, 16])
        # decode block4: torch.Size([b, 512, 32, 32])
        # decode block3: torch.Size([b, 256, 64, 64])
        # decode block2: torch.Size([b, 3, 128, 128])
        # decode block1: torch.Size([b, 3, 256, 256])
        for dec_block, ind, size in zip(self.decoder, reversed(indices), reversed(sizes)):
            x = dec_block[0](x, ind, output_size=size)
            x = dec_block[1:](x)
        return x

# =====================
# 可视化函数
# =====================
def visualize(model, val_loader, save_path, device):
    model.eval()
    images, targets = next(iter(val_loader))
    images = images.to(device)
    targets = (targets.squeeze(1) * 255).long()
    targets[targets == 255] = 0
    with torch.no_grad():
        outputs = model(images)
    predictions = torch.argmax(outputs, dim=1).cpu()

    for i in range(len(images)):
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
        plt.savefig(f"{save_path}_sample_{i}.jpg")
        plt.close()


# =====================
# 主函数入口
# =====================
if __name__ == "__main__":
    # 数据路径与参数
    current_file_path = Path(__file__).resolve()
    current_directory = current_file_path.parent
    os.makedirs(f"{current_directory}/result", exist_ok=True)
    model_path = f"{current_directory}/result/segnet_voc2007_epoch_final.pth"
    data_dir = f"{current_directory}/../01_data/VOC"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 模型定义
    model = SegNet(input_channels=3, num_classes=21).to(device)

    if os.path.exists(model_path):
        print(f"加载已有模型: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("未找到已有模型，从头开始训练。")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader):
            images = images.to(device)
            targets = (targets.squeeze(1) * 255).long().to(device)
            targets[targets == 255] = 0
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader):.4f}")

        if epoch % 5 == 0:
            torch.save(model.state_dict(), model_path)

    # 可视化验证结果
    visualize(model, val_loader, f"{current_directory}/result/segnet_infer", device)
