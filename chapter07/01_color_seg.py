import os
import cv2
import numpy as np

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file_path)
print("当前文件所在目录:", current_dir)
# 创建结果保存目录
save_dir = os.path.join(current_dir, 'result')
os.makedirs(save_dir, exist_ok=True)

# Load the image using OpenCV
image_path = f"{current_dir}/data/color_original_image.jpg"
image = cv2.imread(image_path)
# Convert image from BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Define color range for "green" in RGB
lower_bound = np.array([0, 100, 0])  # Lower bound of green in RGB
upper_bound = np.array([150, 255, 150])  # Upper bound of green in RGB
# Create a mask for the green color
mask = cv2.inRange(image_rgb, lower_bound, upper_bound)
# Apply mask to the image to isolate green regions
segmented_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
segmented_image_save_path = f"{save_dir}/color_segmented_image.jpg"
cv2.imwrite(segmented_image_save_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))  # Save segmented
