import os
import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file_path)
print("当前文件所在目录:", current_dir)
# 创建结果保存目录
save_dir = os.path.join(current_dir, 'result')
os.makedirs(save_dir, exist_ok=True)

# Step 1: Load the image
image_path = f"{current_dir}/data/crf_original_image.jpg"  # 替换为实际图片路径
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
# Step 2: Generate a simulated rough segmentation
# 模拟粗分割，前景为1，背景为0
mask = np.zeros(image.shape[:2], dtype=np.uint8)
mask[118:283, 101:300] = 1  # 中心区域为前景
# Step 3: Prepare data for CRF
labels = mask.flatten()  # 将粗分割标签展平
# Create CRF model
d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)  # 宽 x 高，分类数为2
# Unary potentials (negative log-probability of labels)
unary = unary_from_labels(labels, 2, gt_prob=0.7, zero_unsure=False)
d.setUnaryEnergy(unary)
# Add pairwise Gaussian edge potentials (encourage smoothness)
d.addPairwiseGaussian(sxy=3, compat=10)
# Add pairwise bilateral potentials (edge-aware smoothing)
d.addPairwiseBilateral(sxy=30, srgb=13, rgbim=image, compat=10)
# Step 4: Perform CRF inference
Q = d.inference(50)  # 运行 50 次迭代
result = np.argmax(Q, axis=0).reshape(image.shape[:2])  # 重塑为图像维度
# Step 5: Save results
segmented_image_path = f"{save_dir}/crf_segmented_image.jpg"
# Save CRF-refined segmentation
result_image = (result * 255).astype(np.uint8)  # 转为二值掩码
cv2.rectangle(result_image, (101, 118), (300, 283), 125, 3)
cv2.imwrite(segmented_image_path, result_image)
print(f"Segmented image saved at {segmented_image_path}")
