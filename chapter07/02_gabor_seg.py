import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import convolve

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在的目录
current_dir = os.path.dirname(current_file_path)
print("当前文件所在目录:", current_dir)
# 创建结果保存目录
save_dir = os.path.join(current_dir, 'result')
os.makedirs(save_dir, exist_ok=True)

# Step 1: Load the image using OpenCV
image_path = f"{current_dir}/data/gabor_original_image.jpg"  # Replace with your actual image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")
# Step 2: Generate Gabor filter responses
theta_values = [0, np.pi / 4, np.pi / 2]  # Gabor filter orientations
kernels = [cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F) for theta in theta_values]
gabor_responses = [convolve(image.astype(np.float32), kernel) for kernel in kernels]
# Step 3: Flatten image and Gabor responses for clustering
height, width = image.shape
features = np.stack([response.flatten() for response in gabor_responses], axis=1)
# Step 4: Apply K-means clustering
n_clusters = 2  # Separate rough (grass) and smooth (road) areas
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(features)
# Step 5: Reshape labels back to image dimensions
segmented_result = labels.reshape(height, width)
# Step 6: Prepare the segmented image for visualization and saving
segmented_image = np.zeros((height, width, 3), dtype=np.uint8)
for label in range(n_clusters):
    segmented_image[segmented_result == label] = [label * 255 // (n_clusters - 1)] * 3  # Assign grayscale levels
# Save the segmented image
segmented_image_path = f"{save_dir}/gabor_segmented_image.jpg"
cv2.imwrite(segmented_image_path, segmented_image)
