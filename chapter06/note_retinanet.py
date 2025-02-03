
特征金字塔进行浅层和深层特征融合。
class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs # C3: torch.Size([b, 512, 80, 104]), C4: torch.Size([b, 1024, 40, 52]), C5: torch.Size([b, 2048, 20, 26])

        P5_x = self.P5_1(C5) # Conv2d(2048, 256, (1, 1), (1, 1)), torch.Size([b, 256, 20, 26])
        P5_upsampled_x = self.P5_upsampled(P5_x) # Upsample(scale_factor=2.0, mode='nearest'), torch.Size([b, 256, 40, 52])
        P5_x = self.P5_2(P5_x) # Conv2d(256, 256, (3, 3), (1, 1), (1, 1)), torch.Size([b, 256, 20, 26])

        P4_x = self.P4_1(C4) # Conv2d(1024, 256, (1, 1), (1, 1)), torch.Size([b, 256, 40, 52])
        P4_x = P5_upsampled_x + P4_x # torch.Size([b, 256, 40, 52])
        P4_upsampled_x = self.P4_upsampled(P4_x) # torch.Size([1, 256, 80, 104])
        P4_x = self.P4_2(P4_x) # Conv2d(256, 256, (3, 3), (1, 1),(1, 1)), torch.Size([b, 256, 40, 52])

        P3_x = self.P3_1(C3) # Conv2d(512, 256, (1, 1), (1, 1)), torch.Size([b, 256, 80, 104])
        P3_x = P3_x + P4_upsampled_x # torch.Size([b, 256, 80, 104])
        P3_x = self.P3_2(P3_x) # Conv2d(256, 256, (3, 3), (1, 1), (1, 1)), torch.Size([b, 256, 80, 104])

        P6_x = self.P6(C5) # Conv2d(2048, 256, (3, 3), (2, 2), (1, 1)), torch.Size([b, 256, 10, 13])

        P7_x = self.P7_1(P6_x) # torch.Size([b, 256, 10, 13])
        P7_x = self.P7_2(P7_x) # Conv2d(256, 256, (3, 3), (2, 2), p(1, 1)), torch.Size([b, 256, 5, 7])
        # torch.Size([b, 256, 80, 104]), torch.Size([b, 256, 40, 52]), torch.Size([b, 256, 20, 26]), torch.Size([b, 256, 10, 13]), torch.Size([b, 256, 5, 7])
        return [P3_x, P4_x, P5_x, P6_x, P7_x] # 1/8, 1/16, 1/32, 1/64/ 1/128


# 使用5个3x3卷积获取回归预测结果，特征图每个位置预测9个anchor
# 每个anchor有4个回归参数，torch.Size([b, hxwx9, 4])
# 特征点总数量，80*104+40*52+20*26+10*13+5*7=11085
# 五种输出特征图数量为11085，共预测9种anchor，因此总数量为11085*9=99765。
# 11085, torch.Size([b, 99765, 4])
regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
# 使用5个3x3卷积获取分类预测结果，特征图每个位置预测9个anchor
# 每个anchor预测的数量为类别数量num_cls=80，torch.Size([b, hxwx9, 80])
# 11085, torch.Size([b, 99765, 80])
classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1) 
RegressionModel(
  (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act1): ReLU()
  (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act2): ReLU()
  (conv3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act3): ReLU()
  (conv4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act4): ReLU()
  (output): Conv2d(256, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)
ClassificationModel(
  (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act1): ReLU()
  (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act2): ReLU()
  (conv3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act3): ReLU()
  (conv4): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (act4): ReLU()
  (output): Conv2d(256, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (output_act): Sigmoid()
)

self.ratios = np.array([0.5, 1, 2]) # 三种宽长比
self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]) # 三种尺度
self.sizes = [2 ** (x + 2) for x in self.pyramid_levels] # [32, 64, 128, 256, 512]
def generate_anchors(base_size=16, ratios=None, scales=None):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """
    if ratios is None:
        ratios = np.array([0.5, 1, 2])
    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    num_anchors = len(ratios) * len(scales) # 9
    # initialize output anchors
    anchors = np.zeros((num_anchors, 4)) # (9, 4)
    # scale base_size
    # 将3种尺度下的宽高比1:!时的wh分配到9个anchor
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T 
    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]
    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors
alpha = 0.25
gamma = 2.0
anchor = anchors[0, :, :] # torch.Size([99765, 4])
anchor_widths  = anchor[:, 2] - anchor[:, 0]
anchor_heights = anchor[:, 3] - anchor[:, 1]
anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights
# num_anchors x num_annotations
IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])
# num_anchors x 1
IoU_max, IoU_argmax = torch.max(IoU, dim=1)
targets = torch.ones(classification.shape) * -1 # torch.Size([99765, 80])，中间样本为-1
targets[torch.lt(IoU_max, 0.4), :] = 0 # 小于0.4为负样本
positive_indices = torch.ge(IoU_max, 0.5) # 大于等于0.5为正样本
num_positive_anchors = positive_indices.sum() # 正样本数量
assigned_annotations = bbox_annotation[IoU_argmax, :] # torch.Size([99765, 5])
targets[positive_indices, :] = 0
targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1 # 正样本独热码标签
alpha_factor = torch.ones(targets.shape).cuda() * alpha # torch.Size([99765, 80])，正样本平衡因子
alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor) # 正样本为alpha，其它为1-alpha
focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
focal_weight = alpha_factor * torch.pow(focal_weight, gamma) # torch.Size([99765, 80])，调节因子
bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification)) # torch.Size([99765, 80])
cls_loss = focal_weight * bce # torch.Size([99765, 80])
# 中间样本损失设置为0，即不考虑中间样本
cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda()) # torch.Size([99765, 80])
classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))
# 将坐标转为偏移参数标签
targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
regression_diff = torch.abs(targets - regression[positive_indices, :])
regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                ) # SmoothL1 Loss
regression_losses.append(regression_loss.mean())





