import torch
from torchvision.ops import roi_align
import numpy as np

# 创建一个10x10的特征图，填充一些数值来观察效果
input_feature_map = torch.arange(0.1, 10.1, step=0.1).view(1, 1, 10, 10)  # 1张图, 1通道, 10x10特征图
rois = torch.tensor([[0, 2, 2, 8, 8]], dtype=torch.float32) 
spatial_scale = 1.0   # 特征图和原图尺度相同

output_size = (4, 3)
output_sampling_1 = roi_align(input_feature_map, rois, output_size, spatial_scale, sampling_ratio=1, aligned=False)
print("\nOutput with sampling_ratio=1:\n", output_sampling_1.squeeze().numpy())


output_size = (4, 3)
sampling_ratio = 2 #(每个bin采样2x2个点)
output_sampling_2 = roi_align(input_feature_map, rois, output_size, spatial_scale, sampling_ratio=1, aligned=False)
print("\nOutput with sampling_ratio=2:\n", output_sampling_2.squeeze().numpy())


output_size = (8, 6)

output_sampling_2_2 = roi_align(input_feature_map, rois, output_size, spatial_scale, sampling_ratio=1, aligned=False) # 1x1x8x6
output_sampling_2_2_np = output_sampling_2_2.squeeze().numpy() # 8x6

averaged_output = output_sampling_2_2_np.reshape(4, 2, 3, 2).mean(axis=(1, 3)) # 4x3
print("\noutput_sampling_2_2 with 2x2 pooling:\n", averaged_output)


input_feature_map = torch.arange(0.1, 10.1, step=0.1).view(1, 1, 10, 10)  # 1张图, 1通道, 10x10特征图
rois = torch.tensor([[0, 2, 2, 8, 8]], dtype=torch.float32) 
spatial_scale = 1.0   # 特征图和原图尺度相同
output_size = (4, 3)
output_sampling_align = roi_align(input_feature_map, rois, output_size, spatial_scale, sampling_ratio=1, aligned=True)
print("\nOutput with align:\n", output_sampling_align.squeeze().numpy())
