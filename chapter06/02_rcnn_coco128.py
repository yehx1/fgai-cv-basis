import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import models, transforms
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression


# Step 1: 加载 YOLO 标注数据
def load_yolo_annotations(annotation_file, image_shape):
    annotations = []
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x_center *= image_shape[1]
            y_center *= image_shape[0]
            width *= image_shape[1]
            height *= image_shape[0]
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            annotations.append([int(class_id), x1, y1, x2, y2])
    return annotations

# Step 2: 区域候选生成 (选择性搜索)
def selective_search(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()  # 使用快速模式
    rects = ss.process()
    return rects

class RCNNFeatureExtractor(torch.nn.Module):
    def __init__(self, base_model):
        super(RCNNFeatureExtractor, self).__init__()
        # 包含 AlexNet 的特征提取部分和第一个全连接层（fc6）
        self.features = base_model.features
        self.classifier = torch.nn.Sequential(*list(base_model.classifier.children())[:-3])  # 只保留到 fc6 层

    def forward(self, x): # 1x3x227x227
        x = self.features(x)  # 提取卷积特征 1x256x6x6
        x = torch.flatten(x, 1)  # 展平 1x9216
        x = self.classifier(x)  # 提取 fc6 层特征（4096 维）1x4096
        return x # 1x4096


# Step 4: 定义图像预处理（确保所有图像块调整到 AlexNet 需要的输入尺寸）
transform = transforms.Compose([
    transforms.Resize((227, 227)),  # AlexNet 的输入尺寸是 227x227
    transforms.ToTensor(),
])

# Step 5: 加载图片并生成候选区域
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return np.array(image)

def get_image_regions(image_path):
    image = load_image(image_path)
    rects = selective_search(image)  # 生成候选区域
    return image, rects

# Step 6: 提取候选区域的特征
def extract_features(image, rects, model, device):
    features = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for (x, y, w, h) in rects[:2000]:  # 选前2000个候选区域
            x = max(0, x)
            y = max(0, y)
            w = max(1, w)
            h = max(1, h)
            region = image[y:y+h, x:x+w]
            region_pil = Image.fromarray(region)
            region_tensor = transform(region_pil).unsqueeze(0).to(device) # 1x3x227x227
            feature = model(region_tensor).cpu().numpy().flatten() # 1x4096
            features.append(feature)

    return np.array(features) # 2000x4096

# Step 7: 分类器与回归器（基于 SVM 与线性回归）
def train_svm(features, labels):
    svm_classifier = SVC()
    svm_classifier.fit(features, labels)
    return svm_classifier

def train_regressor(features, bounding_boxes):
    regressor = LinearRegression()
    regressor.fit(features, bounding_boxes)
    return regressor

# 匹配候选区域和真实的 YOLO 边框
def match_rects_with_yolo_annotations(annotations, rects):
    labels = []
    bboxes = []
    matched_features_idx = []

    for idx, rect in enumerate(rects[:2000]):
        x, y, w, h = rect
        rect_x1, rect_y1 = x, y
        rect_x2, rect_y2 = x + w, y + h

        for ann in annotations:
            class_id, x1, y1, x2, y2 = ann

            if iou((rect_x1, rect_y1, rect_x2, rect_y2), (x1, y1, x2, y2)) > 0.5:
                labels.append(class_id)
                bboxes.append([x1, y1, x2, y2])
                matched_features_idx.append(idx)
                break  # 一个候选区域只匹配一个真实标注

    return labels, bboxes, matched_features_idx

# 计算两个矩形的交并比（IoU）
def iou(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x1g, y1g, x2g, y2g = rect2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    rect_area = (x2 - x1) * (y2 - y1)
    gt_area = (x2g - x1g) * (y2g - y1g)
    union_area = rect_area + gt_area - inter_area

    iou = inter_area / union_area if union_area != 0 else 0
    return iou

# 可视化区域和预测结果
def visualize_regions(image, rects, num_regions=10):
    plt.imshow(image)
    for i, (x, y, w, h) in enumerate(rects[:num_regions]):
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))
    plt.show()

# 本程序主要演示RCNN过程，实际训练过程更加复杂。
if __name__ == '__main__':
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    print("当前文件所在目录:", current_dir)
    # 创建结果保存目录
    save_dir = os.path.join(current_dir, 'result')
    os.makedirs(save_dir, exist_ok=True)

    # 获取当前文件的绝对路径
    current_file_path = Path(__file__).resolve()
    # 获取当前文件所在目录
    current_directory = current_file_path.parent

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 1: 加载图片和 YOLO 格式的标注
    image_dir = os.path.join(current_directory, f'{current_dir}/../01_data/coco128/images/')
    annotation_dir = os.path.join(current_directory, f'{current_dir}/../01_data/coco128/labels/')

    image_files = os.listdir(image_dir)
    annotation_files = os.listdir(annotation_dir)

    # 选择一张图片进行示例
    image_filename = image_files[0]
    image_file = os.path.join(image_dir, image_filename)
    annotation_filename = image_filename.replace('.jpg', '.txt')
    annotation_file = os.path.join(annotation_dir, annotation_filename)

    image = load_image(image_file)
    annotations = load_yolo_annotations(annotation_file, image.shape)

    # Step 2: 加载预训练的 AlexNet 模型
    pretrained_model = models.alexnet(pretrained=True)
    feature_extractor = RCNNFeatureExtractor(pretrained_model)

    # Step 3: 生成候选区域
    image, rects = get_image_regions(image_file)
    print('[info] 候选区域总数量：', len(rects))

    # Step 4: 提取候选区域特征
    features = extract_features(image, rects, feature_extractor, device)
    print('[info] features特征维度：', features.shape)

    # Step 5: 匹配候选区域与 YOLO 标注的边界框
    labels, bboxes, matched_features_idx = match_rects_with_yolo_annotations(annotations, rects)

    # 提取匹配到的候选区域的特征
    matched_features = features[matched_features_idx]

    # 将边界框转换为回归器的输入格式
    bboxes = np.array(bboxes)

    # Step 6: 训练 SVM 分类器和线性回归器
    if len(labels) > 0 and len(bboxes) > 0:
        classifier = train_svm(matched_features, labels)
        regressor = train_regressor(matched_features, bboxes)

        # 示例：对候选区域进行分类和边框回归
        predictions = classifier.predict(features)
        predicted_bboxes = regressor.predict(features)

        # 可视化预测结果
        plt.imshow(image)
        cv2.imwrite(f'{save_dir}/rcnn_image.png', image)
        for i, (x, y, w, h) in enumerate(rects[:len(predictions)]):
            if predictions[i]:
                # plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=2))
                x1, y1 = int(x), int(y)
                x2, y2 = x1 + int(w), y1 + int(h)
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # plt.show()
        cv2.imwrite(f'{save_dir}/rcnn_image_1.png', image)
    else:
        print('未找到匹配的候选区域与真实标注，请检查数据。')

    # Step 7: 可视化候选区域
    visualize_regions(image, rects, num_regions=10)
