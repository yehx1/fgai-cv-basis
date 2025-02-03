import os
import cv2
import random
from tqdm import tqdm
import xml.etree.ElementTree as ET
from collections import defaultdict

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

# 用于统计标签类别及目标数量
class_counts = defaultdict(int)
total_objects = 0

# 读取并解析XML文件
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = root.findall("object")
    labels = []
    for obj in objects:
        label = obj.find("name").text
        labels.append(label)
        class_counts[label] += 1
    return labels, objects

# 遍历所有XML文件进行统计
for filename in tqdm(os.listdir(annotations_folder)):
    if filename.endswith(".xml"):
        xml_path = os.path.join(annotations_folder, filename)
        labels, objects = parse_xml(xml_path)
        total_objects += len(objects)

# 输出统计信息
print("各个类别目标数量:")
for label, count in class_counts.items():
    print(f"{label}: {count}")

print(f"总目标数量: {total_objects}")

# 输出
# 各个类别目标数量:
# helmet: 18966
# head: 5785
# person: 751
# 总目标数量: 25502


# 随机选择一个标注文件
annotation_files = [f for f in os.listdir(annotations_folder) if f.endswith(".xml")]
random_annotation_file = random.choice(annotation_files)

# 获取对应的图像文件名称
image_file = random_annotation_file.replace(".xml", ".png")  # 假设图像格式为.jpg
image_path = os.path.join(images_folder, image_file)

# 读取图像
img = cv2.imread(image_path)

# 解析XML文件，获取标注框
tree = ET.parse(os.path.join(annotations_folder, random_annotation_file))
root = tree.getroot()
objects = root.findall("object")

# 在图像上绘制标注框
for obj in objects:
    name = obj.find("name").text
    bndbox = obj.find("bndbox")
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)
    
    # 绘制矩形框并标注类别名称
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.putText(img, name, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# 保存图像
cv2.imwrite(f"{save_dir}/ann_example.jpg", img)

# # 显示图像
# cv2.imshow("Sample Image with Annotations", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()