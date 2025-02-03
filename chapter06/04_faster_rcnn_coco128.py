import os
import cv2
import torch
from PIL import Image
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 自定义数据集类，处理 targets
class CustomCocoDetection(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super(CustomCocoDetection, self).__init__(root, annFile)
        self._transforms = transforms

    def __getitem__(self, index):
        img, target = super(CustomCocoDetection, self).__getitem__(index)

        # 处理 targets，提取 boxes 和 labels
        boxes = []
        labels = []
        for ann in target:
            bbox = ann['bbox']
            x_min, y_min, width, height = bbox
            x_max = x_min + width
            y_max = y_min + height
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'])

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            # 确保 boxes 形状为 [N, 4]
            if boxes.ndim == 1:
                boxes = boxes.unsqueeze(0)
            if labels.ndim == 0:
                labels = labels.unsqueeze(0)
        else:
            # 如果没有目标，创建空张量
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        new_target = {}
        new_target['boxes'] = boxes
        new_target['labels'] = labels

        if self._transforms is not None:
            img, new_target = self._transforms(img, new_target)

        return img, new_target

# 定义同时变换图像和目标的函数
def transform(img, target):
    # 获取原始尺寸
    orig_width, orig_height = img.size

    # 调整图像大小
    img = F.resize(img, (800, 600))

    # 计算缩放比例
    width_ratio = 800 / orig_width
    height_ratio = 600 / orig_height

    if len(target['boxes']) > 0:
        # 确保 boxes 是二维张量
        if target['boxes'].ndim == 1:
            target['boxes'] = target['boxes'].unsqueeze(0)

        # 调整边界框
        target['boxes'][:, [0, 2]] *= width_ratio
        target['boxes'][:, [1, 3]] *= height_ratio

    else:
        # 如果没有目标，创建空张量
        target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)

    # 将图像转换为张量
    img = F.to_tensor(img)

    return img, target

# 训练函数
def train_model(model, data_loader, optimizer, device, save_dir, num_epochs=500):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in data_loader:
            # images: list of Tensors, targets: list of dicts
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 前向传播
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # 反向传播与优化
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(data_loader)}")
        
        # 每 100 个 epoch 保存一次模型
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f'{save_dir}/faster_rcnn_final.pth')
            print(f"Model saved at epoch {epoch+1}")

    # 在训练结束后保存最终模型
    torch.save(model.state_dict(), f'{save_dir}/faster_rcnn_final.pth')
    print("Final model saved.")

# 评估函数
def evaluate_model(model, device, img_path, output_path):
    model.eval()

    # 使用 OpenCV 读取图像
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将 BGR 转换为 RGB，匹配模型输入
    img_pil = Image.fromarray(img_rgb)

    # 获取原始尺寸
    orig_width, orig_height = img_pil.size

    # 调整图像大小
    img_resized = F.resize(img_pil, (800, 600))

    # 转换为张量
    img_tensor = F.to_tensor(img_resized).unsqueeze(0).to(device)

    # 进行预测
    with torch.no_grad():
        prediction = model(img_tensor)

    # 获取预测结果
    boxes = prediction[0]['boxes'].cpu().numpy()  # 边界框
    scores = prediction[0]['scores'].cpu().numpy()  # 置信度
    labels = prediction[0]['labels'].cpu().numpy()  # 类别标签

    # 设定置信度阈值（可以调整）
    score_threshold = 0.1

    # 计算缩放比例（从模型输入大小还原到原始图像大小）
    width_ratio = orig_width / 600
    height_ratio = orig_height / 600

    print('len(boxes): ', len(boxes), boxes, scores)

    # 遍历预测结果，绘制边界框
    for i, box in enumerate(boxes):
        if scores[i] >= score_threshold:
            # 获取边界框的坐标，并还原到原始图像尺寸
            x_min, y_min, x_max, y_max = box
            x_min = int(x_min * width_ratio)
            y_min = int(y_min * height_ratio)
            x_max = int(x_max * width_ratio)
            y_max = int(y_max * height_ratio)

            # 在原始图像上绘制边界框
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # 显示类别标签和置信度
            label = f"{labels[i]}: {scores[i]:.2f}"
            cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存带有预测结果的图像
    cv2.imwrite(output_path, img)
    print(f"Prediction results saved to {output_path}")

if __name__ == '__main__':
    current_file_path = Path(__file__).resolve() # 获取当前文件的绝对路径
    current_directory = current_file_path.parent # 获取当前文件所在目录

    # 创建结果保存目录
    save_dir = os.path.join(current_directory, 'result')
    os.makedirs(save_dir, exist_ok=True)

    # 设置设备
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 数据集路径
    image_dir = f'{current_directory}/../01_data/coco128_coco/images/train2017'
    annotation_file = f'{current_directory}/../01_data/coco128_coco/annotations/instances_train2017.json'

    # 加载自定义的 COCO 数据集
    coco_dataset = CustomCocoDetection(root=image_dir, annFile=annotation_file, transforms=transform)

    # 数据加载器
    # data_loader = DataLoader(coco_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    data_loader = DataLoader(coco_dataset, batch_size=2, shuffle=True, num_workers=4,
                          collate_fn=lambda x: tuple(zip(*x)))
    # 加载预训练的 Faster R-CNN 模型
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # 获取输入特征的维度
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # 替换预训练模型的 box_predictor
    num_classes = 91  # COCO 数据集的类别数，包括背景类。如果您的数据集类别数量不同，请相应修改
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    # 加载模型参数
    if os.path.exists(f'{save_dir}/faster_rcnn_final.pth'):
        print(f'加载已训练模型：{save_dir}/faster_rcnn_final.pth')
        model.load_state_dict(torch.load(f'{save_dir}/faster_rcnn_final.pth'))
    model.to(device)

    # 设置优化器
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    # 训练模型
    train_model(model, data_loader, optimizer, device, save_dir, num_epochs=1000)

    # 评估模型
    test_img_path = f'{image_dir}/000000000081.jpg'
    evaluate_model(model, device, test_img_path, f'{save_dir}/faster_rcnn_image.png')
