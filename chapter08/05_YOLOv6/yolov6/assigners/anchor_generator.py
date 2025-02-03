import torch
from yolov6.utils.general import check_version

torch_1_10_plus = check_version(torch.__version__, minimum='1.10.0')

def generate_anchors(feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5,  device='cpu', is_eval=False, mode='af'):
    '''Generate anchors from features.'''
    anchors = []
    anchor_points = []
    stride_tensor = []
    num_anchors_list = []
    assert feats is not None
    if is_eval:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            shift_x = torch.arange(end=w, device=device) + grid_cell_offset
            shift_y = torch.arange(end=h, device=device) + grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij') if torch_1_10_plus else torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack(
                    [shift_x, shift_y], axis=-1).to(torch.float)
            if mode == 'af': # anchor-free
                anchor_points.append(anchor_point.reshape([-1, 2]))
                stride_tensor.append(
                torch.full(
                    (h * w, 1), stride, dtype=torch.float, device=device))
            elif mode == 'ab': # anchor-based
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3,1))
                stride_tensor.append(
                    torch.full(
                        (h * w, 1), stride, dtype=torch.float, device=device).repeat(3,1))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor
    else:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape # 80, 80
            cell_half_size = grid_cell_size * stride * 0.5 # 20.0， anchor范围
            shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
            shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij') if torch_1_10_plus else torch.meshgrid(shift_y, shift_x) # torch.Size([80, 80])
            anchor = torch.stack(
                [
                    shift_x - cell_half_size, shift_y - cell_half_size,
                    shift_x + cell_half_size, shift_y + cell_half_size
                ],
                axis=-1).clone().to(feats[0].dtype) # anchor尺寸，torch.Size([80, 80, 4])，xyxy
            anchor_point = torch.stack(
                [shift_x, shift_y], axis=-1).clone().to(feats[0].dtype) # torch.Size([80, 80, 2])

            if mode == 'af': # anchor-free
                anchors.append(anchor.reshape([-1, 4])) # torch.Size([6400, 4])
                anchor_points.append(anchor_point.reshape([-1, 2])) # torch.Size([6400, 2])
            elif mode == 'ab': # anchor-based
                anchors.append(anchor.reshape([-1, 4]).repeat(3,1))
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3,1))
            num_anchors_list.append(len(anchors[-1])) # 6400
            stride_tensor.append(
                torch.full(
                    [num_anchors_list[-1], 1], stride, dtype=feats[0].dtype))
        anchors = torch.cat(anchors) # torch.Size([8400, 4])，anchor box坐标，xyxy
        anchor_points = torch.cat(anchor_points).to(device) # torch.Size([8400, 2]), 特征网格中心在原图坐标
        stride_tensor = torch.cat(stride_tensor).to(device) # torch.Size([8400, 1]), 每个锚点对应的缩放尺度stride
        return anchors, anchor_points, num_anchors_list, stride_tensor
