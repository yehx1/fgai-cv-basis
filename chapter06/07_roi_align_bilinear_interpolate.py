import torch

def bilinear_interpolate(feature_map, x1, y1, x2, y2, x, y):
    """
    Performs bilinear interpolation at a specific sampling point (x, y)
    within the region defined by corners (x1, y1) and (x2, y2).
    
    Parameters:
    - feature_map: 2D tensor of feature values.
    - x1, y1: Coordinates of the top-left corner of the region.
    - x2, y2: Coordinates of the bottom-right corner of the region.
    - x, y: Sampling point coordinates within the region.
    
    Returns:
    - Interpolated value at the sampling point (x, y).
    """
    # Values at the four corners
    v11 = feature_map[int(y1), int(x1)]
    v21 = feature_map[int(y1), int(x2)]
    v12 = feature_map[int(y2), int(x1)]
    v22 = feature_map[int(y2), int(x2)]
    print(v11, v21, v12, v22)
    
    # Calculate the distances for interpolation weights
    dx1, dx2 = x - x1, x2 - x
    dy1, dy2 = y - y1, y2 - y

    # Bilinear interpolation formula
    interpolated_value = (v11 * dx2 * dy2 +
                          v21 * dx1 * dy2 +
                          v12 * dx2 * dy1 +
                          v22 * dx1 * dy1)
    
    return interpolated_value

# 示例用法
input_feature_map = torch.arange(0.1, 10.1, step=0.1).view(10, 10)
x1, y1 = 2, 2
x2, y2 = 3, 3
sample_x, sample_y = 3, 2.75  # 采样点在区域内

interpolated_value = bilinear_interpolate(input_feature_map, x1, y1, x2, y2, sample_x, sample_y)
print("Interpolated value at sampling point (x, y):", interpolated_value)

x1, y1 = 2, 3
x2, y2 = 3, 4
sample_x, sample_y = 3, 4.25  # 采样点在区域内

interpolated_value = bilinear_interpolate(input_feature_map, x1, y1, x2, y2, sample_x, sample_y)
print("Interpolated value at sampling point (x, y):", interpolated_value)

input_feature_map = torch.arange(0.1, 10.1, step=0.1).view(10, 10)
x1, y1 = 2, 2
x2, y2 = 3, 3
sample_x, sample_y = 2.5, 2.25  # 采样点在区域内

interpolated_value = bilinear_interpolate(input_feature_map, x1, y1, x2, y2, sample_x, sample_y)
print("Interpolated value at sampling point (x, y):", interpolated_value)

