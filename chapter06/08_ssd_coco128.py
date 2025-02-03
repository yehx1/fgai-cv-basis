import os
import cv2
import torch
from PIL import Image
from pathlib import Path
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
from torchvision.datasets import CocoDetection
from torchvision.models.detection import ssd300_vgg16  # 引入SSD300 VGG16模型

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

            if boxes.ndim == 1:
                boxes = boxes.unsqueeze(0)
            if labels.ndim == 0:
                labels = labels.unsqueeze(0)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        new_target = {'boxes': boxes, 'labels': labels}

        if self._transforms is not None:
            img, new_target = self._transforms(img, new_target)

        return img, new_target

# 定义同时变换图像和目标的函数
def transform(img, target):
    orig_width, orig_height = img.size
    img = F.resize(img, (300, 300))
    width_ratio = 300 / orig_width
    height_ratio = 300 / orig_height

    if len(target['boxes']) > 0:
        if target['boxes'].ndim == 1:
            target['boxes'] = target['boxes'].unsqueeze(0)
        target['boxes'][:, [0, 2]] *= width_ratio
        target['boxes'][:, [1, 3]] *= height_ratio
    else:
        target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)

    img = F.to_tensor(img)
    return img, target

# 训练函数
def train_model(model, data_loader, optimizer, device, save_dir, num_epochs=500):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(data_loader)}")
        
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), f'{save_dir}/ssd300_vgg16.pth')
            print(f"Model saved at epoch {epoch+1}")

    torch.save(model.state_dict(), f'{save_dir}/ssd300_vgg16.pth')
    print("Final model saved.")

# 评估函数
def evaluate_model(model, device, img_path, output_path):
    model.eval()
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    orig_width, orig_height = img_pil.size
    img_resized = F.resize(img_pil, (300, 300))
    img_tensor = F.to_tensor(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)

    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()

    score_threshold = 0.1
    width_ratio = orig_width / 300
    height_ratio = orig_height / 300

    for i, box in enumerate(boxes):
        if scores[i] >= score_threshold:
            x_min, y_min, x_max, y_max = box
            x_min = int(x_min * width_ratio)
            y_min = int(y_min * height_ratio)
            x_max = int(x_max * width_ratio)
            y_max = int(y_max * height_ratio)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            label = f"{labels[i]}: {scores[i]:.2f}"
            cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(output_path, img)
    print(f"Prediction results saved to {output_path}")

if __name__ == '__main__':
    current_file_path = Path(__file__).resolve()
    current_directory = current_file_path.parent

    # 创建结果保存目录
    save_dir = os.path.join(current_directory, 'result')
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    image_dir = f'{current_directory}/../01_data/coco128_coco/images/train2017'
    annotation_file = f'{current_directory}/../01_data/coco128_coco/annotations/instances_train2017.json'

    coco_dataset = CustomCocoDetection(root=image_dir, annFile=annotation_file, transforms=transform)
    data_loader = DataLoader(coco_dataset, batch_size=2, shuffle=True, num_workers=4,
                             collate_fn=lambda x: tuple(zip(*x)))

    model = ssd300_vgg16(pretrained=True)  # 使用SSD300 VGG16模型
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    train_model(model, data_loader, optimizer, device, save_dir, num_epochs=1000)

    test_img_path = f'{image_dir}/000000000081.jpg'
    evaluate_model(model, device, test_img_path, f'{save_dir}/ssd300_vgg16_image.png')
