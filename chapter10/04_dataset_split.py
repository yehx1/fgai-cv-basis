import os
import shutil
import random
from tqdm import tqdm

# 设置路径
images_dir = './data/images/'
labels_dir = './data/yoloann/'
split_parent_dir = './data/helmet_data/'  # 设置划分文件夹的上级目录
split_dirs = ['train', 'val', 'test']

# 检查并删除已有的划分文件夹（直接删除上级目录）
if os.path.exists(split_parent_dir):
    shutil.rmtree(split_parent_dir)

# 创建新的上级目录和子文件夹
os.makedirs(split_parent_dir)
for split_dir in split_dirs:
    os.makedirs(os.path.join(split_parent_dir, split_dir, 'images'))
    os.makedirs(os.path.join(split_parent_dir, split_dir, 'labels'))

# 获取所有图片文件的列表
images = [f for f in os.listdir(images_dir) if f.endswith('.png')]  # 假设图片是.png格式
random.shuffle(images)

# 计算划分的索引
total_images = len(images)
train_size = int(total_images * 0.7)
val_size = int(total_images * 0.2)

# 划分数据集
train_images = images[:train_size]
val_images = images[train_size:train_size + val_size]
test_images = images[train_size + val_size:]

# 将文件复制到相应的文件夹
def copy_files(image_list, split):
    for image in tqdm(image_list):
        label_file = image.replace('.png', '.txt')
        if os.path.exists(os.path.join(labels_dir, label_file)):
            shutil.copy(os.path.join(images_dir, image), os.path.join(split, 'images', image))
            shutil.copy(os.path.join(labels_dir, label_file), os.path.join(split, 'labels', label_file))

# 复制文件到train、val和test文件夹
print("划分训练集……")
copy_files(train_images, os.path.join(split_parent_dir, 'train'))
print("划分验证集……")
copy_files(val_images, os.path.join(split_parent_dir, 'val'))
print("划分测试集……")
copy_files(test_images, os.path.join(split_parent_dir, 'test'))

print("数据集划分完成！")
