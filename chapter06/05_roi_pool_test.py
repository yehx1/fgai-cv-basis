import torch
from torchvision.ops import roi_pool

# 创建一个10x10的特征图，填充一些数值来观察效果
input_feature_map = torch.arange(0.1, 10.1, step=0.1).view(1, 1, 10, 10)  # 1张图, 1通道, 10x10特征图
print("Input feature map:\n", input_feature_map.squeeze().numpy())


# 定义RoI，格式为 [batch_index, x1, y1, x2, y2]
rois = torch.tensor([[0, 2, 2, 8, 8]], dtype=torch.float32)

# 参数设置
output_size = (3, 3)  # 将RoI池化为3x3输出尺寸
spatial_scale = 1.0   # 特征图和原图尺度相同

# 执行 RoI Pooling
output = roi_pool(input_feature_map, rois, output_size, spatial_scale)
print("\nOutput of roi_pool1:\n", output.squeeze().numpy())
# 输出：[[4.5      4.7      4.9]
#       [6.5      6.7      6.9]
#       [8.5      8.7      8.9]]

rois = torch.tensor([[0, 2, 2, 8, 8]], dtype=torch.float32) - 0.49
output = roi_pool(input_feature_map, rois, output_size, spatial_scale)
print("\nOutput of roi_pool2:\n", output.squeeze().numpy())
# 输出：[[4.5      4.7      4.9]
#       [6.5      6.7      6.9]
#       [8.5      8.7      8.9]]

rois = torch.tensor([[0, 2, 2, 8, 8]], dtype=torch.float32) - 0.51
output = roi_pool(input_feature_map, rois, output_size, spatial_scale)
print("\nOutput of roi_pool3:\n", output.squeeze().numpy())
# 输出： [[3.4     3.6       3.8]
#        [5.4       5.6     5.8]
#        [7.4       7.6     7.8]]

rois = torch.tensor([[0, 2, 2, 8, 8]], dtype=torch.float32) 
output_size = (4, 3)
output = roi_pool(input_feature_map, rois, output_size, spatial_scale)
print("\nOutput of roi_pool4:\n", output.squeeze().numpy())
# 输出：  [[3.5       3.7       3.9]
#  [5.5       5.7       5.9]
#  [7.5       7.7       7.9]
#  [8.5       8.7       8.9]]



import torch
from torchvision.ops import roi_pool, roi_align

# 创建一个10x10的特征图，填充一些数值来观察效果
input_feature_map = torch.arange(0.1, 10.1, step=0.1).view(1, 1, 10, 10)  # 1张图, 1通道, 10x10特征图
spatial_scale = 1.0   # 特征图和原图尺度相同
rois = torch.tensor([[0, 2, 2, 8, 8]], dtype=torch.float32) 
output_size = (8, 6)


output_sampling_1 = roi_align(input_feature_map, rois, output_size, spatial_scale, sampling_ratio=1, aligned=False)
print("\nOutput with sampling_ratio=1:\n", output_sampling_1.squeeze().numpy())

# # 2. sampling_ratio = 2 (每个bin采样2x2个点)
# output_sampling_2 = roi_align(input_feature_map, rois, output_size, spatial_scale, sampling_ratio=2, aligned=True)
# print("\nOutput with sampling_ratio=2:\n", output_sampling_2.squeeze().numpy())