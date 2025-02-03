AttributeError: module 'tools.eval' has no attribute 'run'

python tools/train.py


anchors = torch.cat(anchors) # torch.Size([8400, 4])，anchor box坐标，xyxy
anchor_points = torch.cat(anchor_points).to(device) # torch.Size([8400, 2]), 特征网格中心在原图坐标
stride_tensor = torch.cat(stride_tensor).to(device) # torch.Size([8400, 1]), 每个锚点对应的缩放尺度stride
# return anchors, anchor_points, num_anchors_list, stride_tensor

# Intersection area
inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
        (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
# Union Area
w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps
union = w1 * h1 + w2 * h2 - inter + self.eps
iou = inter / union
cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
if self.iou_type == 'giou':
    c_area = cw * ch + self.eps  # convex area
    iou = iou - (c_area - union) / c_area
elif self.iou_type in ['diou', 'ciou']:
    c2 = cw ** 2 + ch ** 2 + self.eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
    if self.iou_type == 'diou':
    iou = iou - rho2 / c2
    elif self.iou_type == 'ciou':
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
            alpha = v / (v - iou + (1 + self.eps))
    iou = iou - (rho2 / c2 + v * alpha)
elif self.iou_type == 'siou':
    # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
    s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + self.eps
    s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + self.eps
    sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
    sin_alpha_1 = torch.abs(s_cw) / sigma
    sin_alpha_2 = torch.abs(s_ch) / sigma
    threshold = pow(2, 0.5) / 2
    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
    angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
    rho_x = (s_cw / cw) ** 2
    rho_y = (s_ch / ch) ** 2
    gamma = angle_cost - 2
    distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
    omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
    omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
    shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
    iou = iou - 0.5 * (distance_cost + shape_cost)
loss = 1.0 - iou

if self.reduction == 'sum':
    loss = loss.sum()
elif self.reduction == 'mean':
    loss = loss.mean()

# return loss

class VarifocalLoss(nn.Module):
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score,gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label # torch.Size([b, 8400, 80])
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight).sum()
        return loss