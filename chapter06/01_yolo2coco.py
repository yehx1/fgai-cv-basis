import os
import cv2
import json
import glob
import shutil
from tqdm import tqdm
from pathlib import Path

def convert_yolo_to_coco(yolo_images_dir, yolo_labels_dir, coco_root_dir):
    """
    将YOLO格式的数据集转换为COCO格式的数据集，并复制图像到COCO目录结构中。

    参数：
    - yolo_images_dir：原始YOLO格式图像的目录。
    - yolo_labels_dir：原始YOLO格式标签的目录。
    - coco_root_dir：目标COCO数据集的根目录。
    """
    # 设置COCO格式的数据集路径
    coco_images_dir = os.path.join(coco_root_dir, 'images/train2017/')
    coco_annotations_dir = os.path.join(coco_root_dir, 'annotations/')

    # 创建COCO格式的目录结构
    os.makedirs(coco_images_dir, exist_ok=True)
    os.makedirs(coco_annotations_dir, exist_ok=True)

    # 创建class_id到category_id的映射和类别列表
    class_id_to_category_id = {}
    categories = []

    category_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
        46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
        67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 84, 85, 86, 87, 88, 89, 90
    ]

    category_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    supercategories = [
        'person', 'vehicle', 'vehicle', 'vehicle', 'vehicle', 'vehicle', 'vehicle',
        'vehicle', 'vehicle', 'outdoor', 'outdoor', 'outdoor', 'outdoor', 'outdoor',
        'animal', 'animal', 'animal', 'animal', 'animal', 'animal', 'animal',
        'animal', 'animal', 'animal', 'accessory', 'accessory', 'accessory',
        'accessory', 'accessory', 'sports', 'sports', 'sports', 'sports', 'sports',
        'sports', 'sports', 'sports', 'sports', 'sports', 'kitchen', 'kitchen',
        'kitchen', 'kitchen', 'kitchen', 'kitchen', 'kitchen', 'food', 'food',
        'food', 'food', 'food', 'food', 'food', 'food', 'food', 'food',
        'furniture', 'furniture', 'furniture', 'furniture', 'furniture',
        'furniture', 'electronic', 'electronic', 'electronic', 'electronic',
        'electronic', 'electronic', 'appliance', 'appliance', 'appliance',
        'appliance', 'appliance', 'indoor', 'indoor', 'indoor', 'indoor', 'indoor',
        'indoor', 'indoor'
    ]

    for idx, (cat_id, name, supercat) in enumerate(zip(category_ids, category_names, supercategories)):
        class_id_to_category_id[idx] = cat_id
        categories.append({'id': cat_id, 'name': name, 'supercategory': supercat})

    images = []
    annotations = []
    annotation_id = 1  # 标注ID起始值
    image_id = 1  # 图像ID起始值

    # 获取所有图像文件（支持.jpg和.png）
    image_files = glob.glob(os.path.join(yolo_images_dir, '*.jpg')) + glob.glob(os.path.join(yolo_images_dir, '*.png'))

    for img_file in tqdm(image_files):
        # 获取文件名和无扩展名的文件名
        filename = os.path.basename(img_file)
        filename_no_ext = os.path.splitext(filename)[0]

        # 读取图像以获取宽度和高度
        img = cv2.imread(img_file)
        if img is None:
            print(f"无法读取图像 {img_file}")
            continue
        height, width = img.shape[:2]

        # 将图像复制到COCO的images/train2017/目录
        coco_image_path = os.path.join(coco_images_dir, filename)
        shutil.copy(img_file, coco_image_path)

        # 创建图像信息
        image_info = {
            'id': image_id,
            'file_name': filename,
            'width': width,
            'height': height
        }
        images.append(image_info)

        # 读取对应的标签文件
        label_file = os.path.join(yolo_labels_dir, filename_no_ext + '.txt')

        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    print(f"{label_file} 中的无效标注：{line}")
                    continue  # 无效的标注
                class_id, x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts)
                class_id = int(class_id)
                if class_id not in class_id_to_category_id:
                    print(f"{label_file} 中的未知类别ID {class_id}")
                    continue
                category_id = class_id_to_category_id[class_id]

                # 将归一化坐标转换为绝对像素坐标
                x_center = x_center_norm * width
                y_center = y_center_norm * height
                bbox_width = width_norm * width
                bbox_height = height_norm * height

                # 将中心点坐标转换为左上角坐标
                x_min = x_center - bbox_width / 2
                y_min = y_center - bbox_height / 2

                # 确保坐标在图像边界内
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                bbox_width = min(width - x_min, bbox_width)
                bbox_height = min(height - y_min, bbox_height)

                # 创建标注信息
                annotation = {
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': category_id,
                    'bbox': [x_min, y_min, bbox_width, bbox_height],
                    'area': bbox_width * bbox_height,
                    'iscrowd': 0,
                    'segmentation': []
                }
                annotations.append(annotation)
                annotation_id += 1
        else:
            # 该图像没有标注
            print(f'[warning] {label_file}不存在。')

        image_id += 1

    # 生成COCO格式的JSON文件
    coco_format = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    # 将标注文件保存到annotations目录，文件名为instances_train2017.json
    json_file_path = os.path.join(coco_annotations_dir, 'instances_train2017.json')
    with open(json_file_path, 'w') as json_file:
        json.dump(coco_format, json_file)

if __name__ == '__main__':
    current_file_path = Path(__file__).resolve() # 获取当前文件的绝对路径
    current_directory = current_file_path.parent # 获取当前文件所在目录

    yolo_images_dir = os.path.join(current_directory, '../01_data/coco128/images/')
    yolo_labels_dir = os.path.join(current_directory, '../01_data/coco128/labels/')
    coco_root_dir = os.path.join(current_directory, '../01_data/coco128_coco/')
    convert_yolo_to_coco(yolo_images_dir, yolo_labels_dir, coco_root_dir)