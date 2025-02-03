https://arxiv.org/abs/2006.04388>`_.
https://zhuanlan.zhihu.com/p/713888154
https://blog.csdn.net/zyw2002/article/details/128732494

line 145
/root/miniconda3/envs/cv/lib/python3.12/site-packages/ultralytics/nn/tasks.py
line 330 380
/root/miniconda3/envs/cv/lib/python3.12/site-packages/ultralytics/engine/trainer.py
line 66
/root/miniconda3/envs/cv/lib/python3.12/site-packages/ultralytics/nn/modules/head.py
line 209 loss
/root/miniconda3/envs/cv/lib/python3.12/site-packages/ultralytics/utils/loss.py

/root/miniconda3/envs/cv/lib/python3.12/site-packages/ultralytics/cfg/models/v8/yolov8.yaml
[0.33, 0.5, 1024]
line 63
/root/miniconda3/envs/cv/lib/python3.12/site-packages/ultralytics/utils/callbacks/tensorboard.py
model(img)

line 805
/root/miniconda3/envs/cv/lib/python3.12/site-packages/ultralytics/engine/model.py
line 475
/root/miniconda3/envs/cv/lib/python3.12/site-packages/torch/jit/_trace.py


def __call__(self, pred_dist, target):
  """
  Return sum of left and right DFL losses.
  Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
  https://ieeexplore.ieee.org/document/9792391
  """
  target = target.clamp_(0, self.reg_max - 1 - 0.01)
  tl = target.long()  # target left, 取整
  tr = tl + 1  # target right
  wl = tr - target  # weight left
  wr = 1 - wl  # weight right
  return (
      F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
      + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
  ).mean(-1, keepdim=True)

# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect

# Parameters
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024] # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
  s: [0.33, 0.50, 1024] # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
  m: [0.67, 0.75, 768] # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
  l: [1.00, 1.00, 512] # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
  x: [1.00, 1.25, 512] # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs

# YOLOv8.0n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 6, C2f, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 3, C2f, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]] # 9



# YOLOv8.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 3, C2f, [512]] # 12

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 3, C2f, [256]] # 15 (P3/8-small)

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]] # cat head P4
  - [-1, 3, C2f, [512]] # 18 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]] # cat head P5
  - [-1, 3, C2f, [1024]] # 21 (P5/32-large)

  - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)


# YOLOv8.0s backbone
backbone: # torch.Size([b, 3, 640, 640])
  # [from, repeats, module, args]
  - [-1, 1, Conv, [32, 3, 2]]  # 0: P1/2, CBS, conv(3, 32, 3, 2), torch.Size([b, 32, 320, 320])
  - [-1, 1, Conv, [64, 3, 2]]  # 1: P2/4, CBS, conv(32, 64, 3, 2), torch.Size([b, 64, 160, 160])
  - [-1, 1, C2f, [64, True]]   # 2: C2f, torch.Size([b, 64, 160, 160])
  - [-1, 1, Conv, [128, 3, 2]] # 3: P3/8, CBS, conv(64, 128, 3, 2), torch.Size([b, 128, 80, 80])
  - [-1, 1, C2f, [128, True]]  # 4: C2f, torch.Size([b, 128, 160, 160])
  - [-1, 1, Conv, [128, 3, 2]] # 5: P4/16, conv(128, 256, 3, 2), torch.Size([b, 256, 40, 40])
  - [-1, 6, C2f, [256, True]]  # 6: C2f, torch.Size([b, 256, 40,40])
  - [-1, 1, Conv, [512, 3, 2]] # 7: P5/32, CBS, conv(256, 512, 3, 2), torch.Size([b, 512, 20, 20])
  - [-1, 3, C2f, [512, True]]  # 8: C2f, torch.Size([b, 512, 20, 20])
  - [-1, 1, SPPF, [1024, 5]]   # 9: SPPF, torch.Size([b, 512, 20, 20])

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 10: Upsample, torch.Size([b, 512, 40, 40]
  - [[-1, 6], 1, Concat, [1]] # 11: cat backbone P4, torch.Size([b, 768, 40,40])
  - [-1, 1, C2f, [256]] # 12: C2f, torch.Size([b, 256, 40, 40])

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 13: Upsample, torch.Size([b, 256, 80, 80]
  - [[-1, 4], 1, Concat, [1]] # 14: cat backbone P3, torch.Size([b, 384, 80, 80]
  - [-1, 3, C2f, [128]] # 15: (P3/8-small), C2f, torch.Size([b, 128, 80, 80])

  - [-1, 1, Conv, [123, 3, 2]] # 16: CBS, conv(128, 128, 3, 2), torch.Size([b, 128, 40, 40])
  - [[-1, 12], 1, Concat, [1]] # 17: cat head P4, torch.Size([b, 384, 40, 40])
  - [-1, 3, C2f, [256]] # 18: (P4/16-medium), C2f, torch.Size([b, 256, 40, 40])

  - [-1, 1, Conv, [256, 3, 2]] # 19: CBS, conv(256, 256, 3, 2), torch.Size([b, 256, 20, 20])
  - [[-1, 9], 1, Concat, [1]]  # 20: cat head P5, torch.Size([b, 768, 20, 20])
  - [-1, 3, C2f, [512]] # 21: (P5/32-large), C2f, torch.Size([b, 512, 20, 20])

  - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4, P5)
80 + 4 * 16
# torch.Size([b, 144, 80, 80]), torch.Size([b, 144, 40, 40]), torch.Size([b, 144, 20, 20])



