import os
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET

# YOLO格式转化函数
def convert_to_yolo_format(xml_file, image_width, image_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = root.findall("object")
    
    yolo_annotations = []
    
    for obj in objects:
        label = obj.find("name").text
        if label not in class_mapping:
            continue  # 只处理 helmet 和 head
        
        class_id = class_mapping[label]
        
        # 获取边界框坐标
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        
        # 计算YOLO格式的标注信息
        x_center = (xmin + xmax) / 2.0 / image_width
        y_center = (ymin + ymax) / 2.0 / image_height
        width = (xmax - xmin) / float(image_width)
        height = (ymax - ymin) / float(image_height)
        
        # 格式化为YOLO标注：<class_id> <x_center> <y_center> <width> <height>
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")
    
    return yolo_annotations

if __name__ == "__main__":
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    print("当前文件所在目录:", current_dir)
    # 创建结果保存目录
    save_dir = os.path.join(current_dir, 'result')
    os.makedirs(save_dir, exist_ok=True)

    # 文件夹路径
    images_folder = "./data/images"
    annotations_folder = "./data/annotations"
    output_folder = "./data/yoloann"  # 输出YOLO标注文件的目录
    os.makedirs(output_folder, exist_ok=True)

    # 类别映射：helmet -> 0, head -> 1
    class_mapping = {"helmet": 0, "head": 1}

    # 遍历所有XML文件并转换为YOLO格式
    for filename in tqdm(os.listdir(annotations_folder)):
        if filename.endswith(".xml"):
            xml_path = os.path.join(annotations_folder, filename)
            
            # 读取对应的图像以获取宽高
            image_file = filename.replace(".xml", ".png")  # 假设图像格式为.jpg
            image_path = os.path.join(images_folder, image_file)
            img = cv2.imread(image_path)
            image_height, image_width, _ = img.shape
            
            # 转换为YOLO格式
            yolo_annotations = convert_to_yolo_format(xml_path, image_width, image_height)
            
            if yolo_annotations:
                # 将YOLO格式标注保存为.txt文件
                yolo_txt_file = os.path.join(output_folder, filename.replace(".xml", ".txt"))
                with open(yolo_txt_file, "w") as f:
                    f.write("\n".join(yolo_annotations))

            # 如果需要，可删除空的标注文件
            if not yolo_annotations:
                print(f"Warning: No relevant objects in {xml_path}, skipping...")
