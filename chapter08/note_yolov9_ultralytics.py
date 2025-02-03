# RevNet

# 训练入口
line 330 380
/root/miniconda3/envs/cv/lib/python3.12/site-packages/ultralytics/engine/trainer.py

# 推理入口
line 145
/root/miniconda3/envs/cv/lib/python3.12/site-packages/ultralytics/nn/tasks.py

# 损失计算入口
line 209
/root/miniconda3/envs/cv/lib/python3.12/site-packages/ultralytics/utils/loss.py

# 配置文件
# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLOv9s object detection model. For Usage examples see https://docs.ultralytics.com/models/yolov9
# 917 layers, 7318368 parameters, 27.6 GFLOPs

# Parameters
nc: 80 # number of classes

# GELAN backbone
backbone:
  - [-1, 1, Conv, [32, 3, 2]] # 0: P1/2, CBS, conv(3, 32, 3, 2), torch.Size([b, 32, 320, 320])
  - [-1, 1, Conv, [64, 3, 2]] # 1: P2/4, CBS, conv(32, 64, 3, 2), torch.Size([b, 64, 160, 160])
  - [-1, 1, ELAN1, [64, 64, 32]] # 2: ELAN1, torch.Size([b, 64, 160, 160])
  - [-1, 1, AConv, [128]] # 3: P3/8, AvgPool(2, 1)+CBS, conv(64, 128, 3, 2), torch.Size([b, 128, 80, 80])
  - [-1, 1, RepNCSPELAN4, [128, 128, 64, 3]] # 4: ELAN (s1, s2, REPCSP, REPCSP)REPCSP=C3+CBS, torch.Size([b, 128, 80, 80]) 
  - [-1, 1, AConv, [192]] # 5-P4/16, AvgPool(2, 1)+CBS, conv(128, 192, 3, 2), torch.Size([b, 192, 40, 40])
  - [-1, 1, RepNCSPELAN4, [192, 192, 96, 3]] # 6: ELAN (s1, s2, REPCSP, REPCSP)REPCSP=C3+CBS, torch.Size([b, 192, 40, 40]) 
  - [-1, 1, AConv, [256]] # 7-P5/32, AvgPool(2, 1)+CBS, conv(192, 256, 3, 2), torch.Size([b, 256, 20, 20])
  - [-1, 1, RepNCSPELAN4, [256, 256, 128, 3]] # 8: RepNCSPELAN4, torch.Size([b, 256, 20, 20])
  - [-1, 1, SPPELAN, [256, 128]] # 9: SPPELAN, torch.Size([b, 256, 20, 20])

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]# 10: Upsample, torch.Size([b, 256, 40, 40]
  - [[-1, 6], 1, Concat, [1]] # 11: cat backbone P4, [10, 6], torch.Size([b, 448, 40, 40]
  - [-1, 1, RepNCSPELAN4, [192, 192, 96, 3]] # 12: RepNCSPELAN4, torch.Size([b, 192, 40, 40]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]# 13: Upsample, torch.Size([b, 192, 80, 80]
  - [[-1, 4], 1, Concat, [1]] # 14: cat backbone P3, [13, 4], torch.Size([b, 320, 80, 80])
  - [-1, 1, RepNCSPELAN4, [128, 128, 64, 3]] # 15: RepNCSPELAN4, torch.Size([b, 128, 80, 80]

  - [-1, 1, AConv, [96]] # 16: AvgPool(2, 1)+CBS, conv(128, 96, 3, 2), torch.Size([b, 96, 40, 40])
  - [[-1, 12], 1, Concat, [1]] # 17: cat head P4, [16, 12], torch.Size([b, 288, 40, 40])
  - [-1, 1, RepNCSPELAN4, [192, 192, 96, 3]] # 18: (P4/16-medium), RepNCSPELAN4, torch.Size([b, 192, 40, 40]

  - [-1, 1, AConv, [128]]# 19: AvgPool(2, 1)+CBS, conv(192, 128, 3, 2), torch.Size([b, 128, 20, 20])
  - [[-1, 9], 1, Concat, [1]] # 20: cat head P5, [19, 9], torch.Size([b, 384, 20, 20])
  - [-1, 1, RepNCSPELAN4, [256, 256, 128, 3]] # 21: (P5/32-large), RepNCSPELAN4, torch.Size([b, 256, 20, 20]
  # torch.Size([b, 128, 80, 80], torch.Size([b, 192, 40, 40], torch.Size([b, 256, 20, 20]
  - [[15, 18, 21], 1, Detect, [nc]] # Detect(P3, P4 P5)
  # torch.Size([b, 144, 80, 80]), torch.Size([b, 144, 40, 40]), torch.Size([b, 144, 20, 20])


