import torch
from torchvision.ops import nms as torchvision_nms

def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations using PyTorch's built-in NMS."""

    if dets.shape[0] == 0:
        return torch.empty((0,), dtype=torch.int64)
    
    boxes = dets[:, :4]  # Assuming dets format is [x1, y1, x2, y2, score]
    scores = dets[:, 4]

    if force_cpu or not torch.cuda.is_available():
        # Use CPU for NMS
        return torchvision_nms(boxes, scores, thresh)
    else:
        # Use GPU for NMS if CUDA is available
        return torchvision_nms(boxes.to('cuda'), scores.to('cuda'), thresh)
