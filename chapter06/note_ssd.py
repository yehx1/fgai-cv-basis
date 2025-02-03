"D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],


anchor_generator = DefaultBoxGenerator(
    [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
    steps=[8, 16, 32, 64, 100, 300],
)
sk = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
steps=[8, 16, 32, 64, 100, 300]
def _generate_wh_pairs(self, num_outputs)
    _wh_pairs: List[Tensor] = []
    for k in range(num_outputs):
        # scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
        # 300*scales=[21, 45, 99, 153, 207, 261, 315]
        s_k = self.scales[k]
        s_prime_k = math.sqrt(self.scales[k] * self.scales[k + 1])
        wh_pairs = [[s_k, s_k], [s_prime_k, s_prime_k]]
        # Adding 2 pairs for each aspect ratio of the feature map k
        for ar in self.aspect_ratios[k]:
            sq_ar = math.sqrt(ar)
            w = self.scales[k] * sq_ar
            h = self.scales[k] / sq_ar
            wh_pairs.extend([[w, h], [h, w]])
        _wh_pairs.append(torch.as_tensor(wh_pairs, dtype=dtype, device=device))
    return _wh_pairs

# images: torch.Size([b, 3, 300, 300])
images, targets = self.transform(images, targets)
# vgg16 + extra layers
# torch.Size([b, 512, 38, 38]), torch.Size([b, 1024, 19, 19]), torch.Size([b, 512, 10, 10])
# torch.Size([b, 256, 5, 5]), torch.Size([b, 256, 3, 3]), torch.Size([b, 256, 1, 1])
features = self.backbone(images.tensors)
# bbox_regression: torch.Size([b, 8732, 4])
# cls_logits: torch.Size([b, 8732, 91])
head_outputs = self.head(features)
# [torch.Size([8732, 4])]: 对应原图anchor位置x1, y1, x2, y2
anchors = self.anchor_generator(images, features)
# 计算损失
losses = self.compute_loss(targets, head_outputs, anchors, matched_idxs)
# 推理阶段阈值处理、NMS非极大值预处理、尺寸恢复等操作
detections = self.postprocess_detections(head_outputs, anchors, images.image_sizes)
detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU(inplace=True)
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (6): ReLU(inplace=True)
  (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (8): ReLU(inplace=True)
  (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (11): ReLU(inplace=True)
  (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (13): ReLU(inplace=True)
  (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (15): ReLU(inplace=True)
  (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
  (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (18): ReLU(inplace=True)
  (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (20): ReLU(inplace=True)
  (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (22): ReLU(inplace=True)
)


Conv3-64，Conv3-64，最大池化（2x2）：torch.Size([b, 64, 150, 150])
Conv3-128，Conv3-128，最大池化（2x2）：torch.Size([b, 128, 75, 75])
Conv3-256，Conv3-256，Conv3-256，最大池化（2x2）：torch.Size([b, 256, 38, 38])
Conv3-512，Conv3-512，Conv3-512：torch.Size([b, 512, 38, 38])
Conv3-512，Conv3-512，Conv3-512，最大池化（2x2）：
全连接层（4096神经元，ReLU）：
全连接层（4096神经元，ReLU）：
全连接层（1000神经元，Softmax）：

# torch.Size([b, 512, 38, 38]), Conv4_3
# torch.Size([b, 1024, 19, 19]), Conv7
# torch.Size([b, 512, 10, 10]), Conv8_2
# torch.Size([b, 256, 5, 5]), Conv9_2
# torch.Size([b, 256, 3, 3]), Conv10_2
# torch.Size([b, 256, 1, 1]), Conv11_2
ModuleList(
  (0): Sequential(
    (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (2): ReLU(inplace=True)
    (3): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): ReLU(inplace=True)
    # Conv5_3
    (5): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace=True)
    (7): Sequential(
      (0): MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
      # Conv6
      (1): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6))
      (2): ReLU(inplace=True)
      # Conv7
      (3): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
      (4): ReLU(inplace=True)
    )
  )
  (1): Sequential(
    (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU(inplace=True)
    # Conv8_2
    (2): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (3): ReLU(inplace=True)
  )
  (2): Sequential(
    (0): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU(inplace=True)
    # Conv9_2
    (2): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (3): ReLU(inplace=True)
  )
  (3): Sequential(
    (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU(inplace=True)
    # Conv10_2
    (2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
    (3): ReLU(inplace=True)
  )
  (4): Sequential(
    (0): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
    (1): ReLU(inplace=True)
    # Conv11_2
    (2): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1))
    (3): ReLU(inplace=True)
  )
)

head_outputs = self.head(features)
# bbox_regression: torch.Size([b, 8732, 4])
# cls_logits: torch.Size([b, 8732, 91])
# 6种尺度特征图的每个特征点对应的anchor数量为[4, 6, 6, 6, 4, 4]
# 回归每个位置预测4个参数，因而输出通道数量分别为[16, 24, 24, 24, 16, 16]
# 分类每个位置预测类别，类别数量为91，因而输出通道数量分别为[364, 546, 546, 546,, 364, 364]
# 特征点位置总数量为：38*38*4+19*19*6+10*10*6+5*5*6+3*3*4+1*4 = 8732
"bbox_regression": self.regression_head(x)
SSDRegressionHead(
  (module_list): ModuleList(
    (0): Conv2d(512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): Conv2d(1024, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (2): Conv2d(512, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): Conv2d(256, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): Conv2d(256, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
)
"cls_logits": self.classification_head(x)
SSDClassificationHead(
  (module_list): ModuleList(
    (0): Conv2d(512, 364, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): Conv2d(1024, 546, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (2): Conv2d(512, 546, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): Conv2d(256, 546, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): Conv2d(256, 364, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): Conv2d(256, 364, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
)


cls_loss = F.cross_entropy(cls_logits, cls_targets)
# Hard Negative Sampling
foreground_idxs = cls_targets > 0 
num_negative = self.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True) # 负样本3：1
# num_negative[num_negative < self.neg_to_pos_ratio] = self.neg_to_pos_ratio
negative_loss = cls_loss.clone() # 把背景预测成目标
negative_loss[foreground_idxs] = -float("inf")  # use -inf to detect positive values that creeped in the sample
values, idx = negative_loss.sort(1, descending=True) # 损失越大，说明预测结果越错误，越困难
# background_idxs = torch.logical_and(idx.sort(1)[1] < num_negative, torch.isfinite(values))
background_idxs = idx.sort(1)[1] < num_negative # 相当于记录negative每个值的排名
N = max(1, num_foreground)
loss: "classification": (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N
