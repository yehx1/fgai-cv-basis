import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage import color, exposure

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file_path)
print("当前文件所在目录:", current_dir)
# 创建结果保存目录
save_dir = os.path.join(current_dir, 'result')
os.makedirs(save_dir, exist_ok=True)

# Step 1: Load the image using OpenCV
image_path = f"{current_dir}/data/hog_original_image.jpg"  # Replace with your actual image path
image = cv2.imread(image_path)  # Load the image in BGR format
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
# Convert image to RGB as HOG expects RGB format
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Step 2: Convert image to grayscale
gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
# Step 3: Compute HOG features and visualization
hog_features, hog_image = hog(
    gray_image,
    orientations=9,               # Number of gradient orientations
    pixels_per_cell=(8, 8),       # Size of a cell
    cells_per_block=(2, 2),       # Number of cells per block
    visualize=True,               # Generate the visualization
    channel_axis=None             # Since the image is grayscale, channel_axis=None
)
# Step 4: Enhance the HOG visualization for better visibility
hog_image_rescaled = exposure.equalize_hist(hog_image)  # Equalize histogram for better contrast
# Step 5: Convert to 8-bit format and save the visualization
hog_image_rescaled_8bit = (hog_image_rescaled * 255).astype(np.uint8)
hog_visualization_path = f"{save_dir}/hog_segmented_image.jpg"
cv2.imwrite(hog_visualization_path, hog_image_rescaled_8bit)  # Save the HOG visualization