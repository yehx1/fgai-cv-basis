DarkNet53BackBone(
  (conv1): ConvLayer(
    (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (lrelu): LeakyReLU(negative_slope=0.1)
  )
  (cr_block1): Sequential(
    (conv): ConvLayer(
      (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (lrelu): LeakyReLU(negative_slope=0.1)
    )
    (res0): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
  )
  (cr_block2): Sequential(
    (conv): ConvLayer(
      (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (lrelu): LeakyReLU(negative_slope=0.1)
    )
    (res0): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (res1): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
  )
  (cr_block3): Sequential(
    (conv): ConvLayer(
      (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (lrelu): LeakyReLU(negative_slope=0.1)
    )
    (res0): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (res1): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (res2): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (res3): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (res4): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (res5): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (res6): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (res7): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
  )
  (cr_block4): Sequential(
    (conv): ConvLayer(
      (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (lrelu): LeakyReLU(negative_slope=0.1)
    )
    (res0): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (res1): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (res2): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (res3): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (res4): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (res5): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (res6): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (res7): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
  )
  (cr_block5): Sequential(
    (conv): ConvLayer(
      (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (lrelu): LeakyReLU(negative_slope=0.1)
    )
    (res0): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (res1): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (res2): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
    (res3): ResBlock(
      (conv1): ConvLayer(
        (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
      (conv2): ConvLayer(
        (conv): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (lrelu): LeakyReLU(negative_slope=0.1)
      )
    )
  )
)


class DarkNet53BackBone(nn.Module):
  def __init__(self):
      super(DarkNet53BackBone, self).__init__()
      self.conv1 = ConvLayer(3, 32, 3)
      self.cr_block1 = make_conv_and_res_block(32, 64, 1)
      self.cr_block2 = make_conv_and_res_block(64, 128, 2)
      self.cr_block3 = make_conv_and_res_block(128, 256, 8)
      self.cr_block4 = make_conv_and_res_block(256, 512, 8)
      self.cr_block5 = make_conv_and_res_block(512, 1024, 4)
  def forward(self, x): # torch.Size([b, 3, 416, 416])
      tmp = self.conv1(x) # CBL, torch.Size([b, 32, 416, 416])
      tmp = self.cr_block1(tmp) # res1, torch.Size([b, 64, 208, 208])
      tmp = self.cr_block2(tmp) # res2, torch.Size([b, 128, 104, 104])
      out3 = self.cr_block3(tmp) # res8, torch.Size([b, 256, 52, 52])
      out2 = self.cr_block4(out3) # res8, torch.Size([b, 512, 26, 26])
      out1 = self.cr_block5(out2) # res4, torch.Size([b, 1024, 13, 13])
      return out1, out2, out3

# Neck + Head
class YoloNetTail(nn.Module):
  def __init__(self):
      super(YoloNetTail, self).__init__()
      self.detect1 = DetectionBlock(1024, 1024, 'l', 32)
      self.conv1 = ConvLayer(512, 256, 1)
      self.detect2 = DetectionBlock(768, 512, 'm', 16)
      self.conv2 = ConvLayer(256, 128, 1)
      self.detect3 = DetectionBlock(384, 256, 's', 8)
  def forward(self, x1, x2, x3):
      out1 = self.detect1(x1) # torch.Size([b, 13*13*3, 85])
      branch1 = self.detect1.branch # torch.Size([b, 512, 13, 13])
      tmp = self.conv1(branch1) # CBL, torch.Size([b, 256, 13, 13])
      tmp = F.interpolate(tmp, scale_factor=2) # torch.Size([b, 256, 26, 26])
      tmp = torch.cat((tmp, x2), 1) # concat, torch.Size([b, 768, 26, 26])
      out2 = self.detect2(tmp) # torch.Size([b, 26*26*3, 85])
      branch2 = self.detect2.branch # torch.Size([b, 256, 26, 26])
      tmp = self.conv2(branch2) # torch.Size([b, 128, 26, 26])
      tmp = F.interpolate(tmp, scale_factor=2) # torch.Size([b, 128, 52, 52])
      tmp = torch.cat((tmp, x3), 1) # torch.Size([b, 384, 52, 52])
      out3 = self.detect3(tmp) # torch.Size([b, 52*52*3, 85])
      return out1, out2, out3

class YoloNetV3(nn.Module):
  def __init__(self, nms=False, post=True):
      super(YoloNetV3, self).__init__()
      self.darknet = DarkNet53BackBone()
      self.yolo_tail = YoloNetTail()
      self.nms = nms
      self._post_process = post
  def forward(self, x):
      tmp1, tmp2, tmp3 = self.darknet(x) # tmp1: torch.Size([b, 1024, 13, 13]), tmp2: torch.Size([b, 512, 26, 26]), tmp3: torch.Size([b, 256, 52, 52])
      out1, out2, out3 = self.yolo_tail(tmp1, tmp2, tmp3) # out1: torch.Size([b, 13*13*3, 85]), out2: torch.Size([b, 26*26*3, 85]), out3: torch.Size([b, 52*52*3, 85])
      out = torch.cat((out1, out2, out3), 1) # torch.Size([b, 10647, 85])
      logging.debug("The dimension of the output before nms is {}".format(out.size()))
      return out # torch.Size([b, 10647, 85])

真实标签计算
def pre_process_targets(tgt: Tensor, tgt_len, img_size):
    """get the index of the predictions corresponding to the targets;
    and put targets from different sample into one dimension (flatten), getting rid of the tails;
    and convert coordinates to local.
    Args:
        tgt: (tensor) the tensor of ground truths (targets). Size is [B, N_tgt_max, NUM_ATTRIB].
                    where B is the batch size;
                    N_tgt_max is the max number of targets in this batch;
                    NUM_ATTRIB is the number of attributes, determined in config.py.
                    coordinates is in format cxcywh and is global.
                    If a certain sample has targets fewer than N_tgt_max, zeros are filled at the tail.
        tgt_len: (Tensor) a 1D tensor showing the number of the targets for each sample. Size is [B, ].
        img_size: (int) the size of the training image.
    :return
        tgt_t_flat: (tensor) the flattened and local target. Size is [N_tgt_total, NUM_ATTRIB],
                            where N_tgt_total is the total number of targets in this batch.
        idx_obj_1d: (tensor) the tensor of the indices of the predictions corresponding to the targets.
                            The size is [N_tgt_total, ]. Note the indices have been added the batch number,
                            therefore when the predictions are flattened, the indices can directly find the prediction.
    """
    # find the anchor box which has max IOU (zero centered) with the targets
    wh_anchor = torch.tensor(ANCHORS).to(tgt.device).float() # 9种尺寸Anchor，torch.Size([9, 2])
    n_anchor = wh_anchor.size(0) # 9
    xy_anchor = torch.zeros((n_anchor, 2), device=tgt.device) # torch.Size([9, 2])
    bbox_anchor = torch.cat((xy_anchor, wh_anchor), dim=1) # torch.Size([9, 4])
    bbox_anchor.unsqueeze_(0) # torch.Size([1, 9, 4])
    iou_anchor_tgt = iou_batch(bbox_anchor, tgt[..., :4], zero_center=True) # torch.Size([b, 9, K]), 真实标签与anchor的中心均为网格左上角坐标，相当于均为(0, 0)
    _, idx_anchor = torch.max(iou_anchor_tgt, dim=1) # torch.Size([8, K]), 每个真实框对应的anchor索引

    # find the corresponding prediction's index for the anchor box with the max IOU
    strides_selection = [8, 16, 32]
    scale = idx_anchor // 3 # torch.Size([8, K]), 属于几种尺寸特征图
    idx_anchor_by_scale = idx_anchor - scale * 3 #  # torch.Size([8, K]), 属于指定特征图中的第几种anchor
    stride = 8 * 2 ** scale #  # torch.Size([8, K])， 相对于输入的缩放倍数
    grid_x = (tgt[..., 0] // stride.float()).long() # torch.Size([8, K])，真实框匹配的Anchor所在特征网格的左上角坐标
    grid_y = (tgt[..., 1] // stride.float()).long() # torch.Size([8, K])，真实框匹配的Anchor所在特征网格的左上角坐标
    n_grid = img_size // stride # torch.Size([8, K]), 对应特征图尺寸
    large_scale_mask = (scale <= 1).long() # 大尺度特征图，输出特征图尺寸是有小变大，与Anchor大小顺序相反
    med_scale_mask = (scale <= 0).long() # 中尺度特征图，在输出完大目标（小尺寸特征图）之后
    idx_obj = \ 
        large_scale_mask * (img_size // strides_selection[2]) ** 2 * 3 + \
        med_scale_mask * (img_size // strides_selection[1]) ** 2 * 3 + \
        n_grid ** 2 * idx_anchor_by_scale + n_grid * grid_y + grid_x # 计算真实目标所属Anchor的索引

    # calculate t_x and t_y
    t_x = (tgt[..., 0] / stride.float() - grid_x.float()).clamp(EPSILON, 1 - EPSILON)
    t_x = torch.log(t_x / (1. - t_x))   #inverse of sigmoid
    t_y = (tgt[..., 1] / stride.float() - grid_y.float()).clamp(EPSILON, 1 - EPSILON)
    t_y = torch.log(t_y / (1. - t_y))    # inverse of sigmoid

    # calculate t_w and t_h
    w_anchor = wh_anchor[..., 0]
    h_anchor = wh_anchor[..., 1]
    w_anchor = torch.index_select(w_anchor, 0, idx_anchor.view(-1)).view(idx_anchor.size())
    h_anchor = torch.index_select(h_anchor, 0, idx_anchor.view(-1)).view(idx_anchor.size())
    t_w = torch.log((tgt[..., 2] / w_anchor).clamp(min=EPSILON))
    t_h = torch.log((tgt[..., 3] / h_anchor).clamp(min=EPSILON))

    # the raw target tensor
    tgt_t = tgt.clone().detach()

    tgt_t[..., 0] = t_x
    tgt_t[..., 1] = t_y
    tgt_t[..., 2] = t_w
    tgt_t[..., 3] = t_h

    # aggregate processed targets and the corresponding prediction index from different batches in to one dimension
    n_batch = tgt.size(0)
    n_pred = sum([(img_size // s) ** 2 for s in strides_selection]) * 3 # 10647

    idx_obj_1d = []
    tgt_t_flat = []

    for i_batch in range(n_batch):
        v = idx_obj[i_batch] # 真实框匹配的Anchor索引, K
        t = tgt_t[i_batch] # torch.Size([K, 85])，编码为预测值的标签
        l = tgt_len[i_batch] # 真实目标数量
        idx_obj_1d.append(v[:l] + i_batch * n_pred) # 展平后，全部真实框对应的Anchor索引
        tgt_t_flat.append(t[:l]) # 实际数量经过编码后的标签，[tx, ty, tw, th, 1.0, onehot-class]

    idx_obj_1d = torch.cat(idx_obj_1d) # 展平后，全部真实框对应的Anchor索引
    tgt_t_flat = torch.cat(tgt_t_flat) # 经过预测编码后的实际数量标签，[tx, ty, tw, th, 1.0, onehot-class]

    return tgt_t_flat, idx_obj_1d










def encoder(self,boxes,labels):
    '''
    boxes (tensor) [[x1,y1,x2,y2],[]]
    labels (tensor) [...]
    return 7x7x30
    '''
    grid_num = 7
    target = torch.zeros((grid_num,grid_num,30)) # torch.Size([14, 14, 30])
    cell_size = 1./grid_num
    wh = boxes[:,2:]-boxes[:,:2]
    cxcy = (boxes[:,2:]+boxes[:,:2])/2
    for i in range(cxcy.size()[0]):
        cxcy_sample = cxcy[i]
        ij = (cxcy_sample/cell_size).ceil()-1 # 中心点的网格坐标
        target[int(ij[1]),int(ij[0]),4] = 1 # 置信度为1
        target[int(ij[1]),int(ij[0]),9] = 1 # 置信度为1
        target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1 # label，独热码
        xy = ij*cell_size #匹配到的网格的左上角相对坐标
        # 对于真实框，两个候选框的预测目标一致
        # 中心点相对于网格的偏移, 特征图上绝对坐标偏移
        delta_xy = (cxcy_sample -xy)/cell_size
        target[int(ij[1]),int(ij[0]),2:4] = wh[i] # 第一个候选框宽度和长度比例
        target[int(ij[1]),int(ij[0]),:2] = delta_xy # 第一个候选框中心偏移
        target[int(ij[1]),int(ij[0]),7:9] = wh[i] # 第二个候选框宽度和长度比例
        target[int(ij[1]),int(ij[0]),5:7] = delta_xy # 第二个候选框中心偏移
    return target


def forward(self,pred_tensor,target_tensor):
'''
pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
target_tensor: (tensor) size(batchsize,S,S,30)
'''
N = pred_tensor.size()[0]
coo_mask = target_tensor[:,:,:,4] > 0
noo_mask = target_tensor[:,:,:,4] == 0
coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor) # torch.Size([b, 14, 14, 30])
noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor) # torch.Size([b, 14, 14, 30])
coo_pred = pred_tensor[coo_mask].view(-1,30) # 有目标处的预测结果，torch.Size([N, 30])
box_pred = coo_pred[:,:10].contiguous().view(-1,5) # box[x1,y1,w1,h1,c1], torch.Size([2N, 5])
class_pred = coo_pred[:,10:] # 类别预测, torch.Size(N,20])
coo_target = target_tensor[coo_mask].view(-1,30) # torch.Size([N, 30])
box_target = coo_target[:,:10].contiguous().view(-1,5) # torch.Size([2N, 5])
class_target = coo_target[:,10:] # torch.Size([N, 20])
# compute not contain obj loss
noo_pred = pred_tensor[noo_mask].view(-1,30) # 没有目标的点，torch.Size([K, 30])
noo_target = target_tensor[noo_mask].view(-1,30) # torch.Size([K, 30])
noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size()) # torch.Size([K, 30])
noo_pred_mask.zero_()
noo_pred_mask[:,4]=1;noo_pred_mask[:,9]=1
noo_pred_c = noo_pred[noo_pred_mask] # noo pred只需要计算c的损失 size[-1,2]，只计算有无目标，torch.Size([2k])
noo_target_c = noo_target[noo_pred_mask] # torch.Size([2k]，真实目标标签均为背景标签0
nooobj_loss = F.mse_loss(noo_pred_c,noo_target_c,size_average=False) # 背景损失，无目标置信度损失
#compute contain obj loss
coo_response_mask = torch.cuda.ByteTensor(box_target.size()) # torch.Size([2N, 5])
coo_response_mask.zero_()
coo_not_response_mask = torch.cuda.ByteTensor(box_target.size())
coo_not_response_mask.zero_() # torch.Size([2N, 5]
box_target_iou = torch.zeros(box_target.size()).cuda() # torch.Size([2K, 5]
for i in range(0,box_target.size()[0],2): #choose the best iou box
    box1 = box_pred[i:i+2]
    box1_xyxy = Variable(torch.FloatTensor(box1.size()))
    box1_xyxy[:,:2] = box1[:,:2]/14. - 0.5*box1[:,2:4]
    box1_xyxy[:,2:4] = box1[:,:2]/14. + 0.5*box1[:,2:4]
    box2 = box_target[i].view(-1,5)
    box2_xyxy = Variable(torch.FloatTensor(box2.size()))
    box2_xyxy[:,:2] = box2[:,:2]/14. - 0.5*box2[:,2:4]
    box2_xyxy[:,2:4] = box2[:,:2]/14. + 0.5*box2[:,2:4]
    iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]
    max_iou,max_index = iou.max(0) # torch.Size([2, 1])
    max_index = max_index.data.cuda() # 选择IOU较大检测框
    coo_response_mask[i+max_index]=1 # 两个预测框中跟相关的
    coo_not_response_mask[i+1-max_index]=1 # 两个预测框中不想管的
    box_target_iou[i+max_index,torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
box_target_iou = Variable(box_target_iou).cuda() # torch.Size([2N, 5])
#1.response loss
box_pred_response = box_pred[coo_response_mask].view(-1,5 ) # torch.Size([N, 5])
box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5) # torch.Size([N, 5])
box_target_response = box_target[coo_response_mask].view(-1,5) # torch.Size([N, 5])
contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False) # 包含目标损失
loc_loss = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],size_average=False) + F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),size_average=False) # 位置损失
#2.not response loss
box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5) # torch.Size([N, 5])
box_target_not_response = box_target[coo_not_response_mask].view(-1,5) # torch.Size([N, 5])
box_target_not_response[:,4]= 0 # 将剩余B-1个位置的目标设置为背景，即仅有一个有效的预测框
# 确保B个结果只有一个包含目标，实际上是不包含目标损失的一部分
not_contain_loss = F.mse_loss(box_pred_not_response[:,4], box_target_not_response[:,4],size_average=False)
#3.class loss
class_loss = F.mse_loss(class_pred,class_target,size_average=False) # 分类损失
return (self.l_coord*loc_loss + 2*contain_loss + not_contain_loss + self.l_noobj*nooobj_loss + class_loss)/N
