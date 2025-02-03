https://blog.csdn.net/weixin_43694096/article/details/136992990
git clone https://github.com/WongKinYiu/yolov9.git

https://cloud.tencent.com/developer/article/2390383
https://github.com/WongKinYiu/yolov9/issues/192


train.py
# 关闭wandb
<yolov9_dir>/utils/loggers/__init__.py
line 31: wandb = None
# 训练入口
<yolov9_dir>/train.py
line 303: pred = model(imgs)

# 推理入口
<yolo_dir>/models/yolo.py
line 527: y, dt = [], []

# 损失函数入口
<yolov9_dir>/train.py
line 304: loss, loss_items = compute_loss
<yolov9_dir>/utils/loss_tal.py
line 166: loss = torch.zeros(3, device=self.device)


train_dual.py
# 训练入口
<yolov9_dir>/train_dual.py
line 314: pred = model(imgs)

# 推理入口
<yolo_dir>/models/yolo.py
line 527: y, dt = [], []

# 损失函数入口
<yolov9_dir>/train.py
line 315: loss, loss_items = compute_loss
<yolov9_dir>/utils/loss_tal_dual.py
line 171: loss = torch.zeros(3, device=self.device)



train_triple.py
# 训练入口
<yolov9_dir>/train_dual.py
line 308: pred = model(imgs)

# 推理入口
<yolo_dir>/models/yolo.py
line 527: y, dt = [], []

# 损失函数入口
<yolov9_dir>/train.py
line 309: loss, loss_items = compute_loss
<yolov9_dir>/utils/loss_tal_dual.py
line 171: loss = torch.zeros(3, device=self.device)



backbone:
  [
   # conv down
   [-1, 1, Conv, [32, 3, 2]],   # 0: input->0, P1/2, CBS, conv(3, 32, 3, 2), torch.Size([b, 32, 320, 320])
   # conv down
   [-1, 1, Conv, [64, 3, 2]],   # 1: 0->1, P2/4, CBS, conv(32, 64, 3, 2), torch.Size([b, 64, 160, 160])
   # elan-1 block
   [-1, 1, ELAN1, [64, 64, 32]],# 2: 1->2, ELAN1, torch.Size([b, 64, 160, 160])
   # avg-conv down
   [-1, 1, AConv, [128]],       # 3: 2->3, P3/8, AvgPool(2, 1)+CBS, conv(64, 128, 3, 2), torch.Size([b, 128, 80, 80])
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [128, 128, 64, 3]],# 4: 3->4, ELAN (s1, s2, REPCSP, REPCSP)REPCSP=C3+CBS, torch.Size([b, 128, 80, 80]) 
   # avg-conv down
   [-1, 1, AConv, [192]],  # 5: 4->5, P4/16, AvgPool(2, 1)+CBS, conv(128, 192, 3, 2), torch.Size([b, 192, 40, 40])
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [192, 192, 96, 3]],  # 6: 5->6, ELAN (s1, s2, REPCSP, REPCSP)REPCSP=C3+CBS, torch.Size([b, 192, 40, 40]) 
   # avg-conv down
   [-1, 1, AConv, [256]],  # 7: 6->7, P5/32, AvgPool(2, 1)+CBS, conv(192, 256, 3, 2), torch.Size([b, 256, 20, 20])
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [256, 256, 128, 3]],  # 8: 7->8, RepNCSPELAN4, torch.Size([b, 256, 20, 20])
  ]

# elan head
head:
  [
   # elan-spp block
   [-1, 1, SPPELAN, [256, 128]],  # 9: 8->9, SPPELAN, torch.Size([b, 256, 20, 20])
   # up-concat merge
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],# 10: 9->10, Upsample, torch.Size([b, 256, 40, 40]
   [[-1, 6], 1, Concat, [1]],  # 11: cat backbone P4, [10, 6], torch.Size([b, 448, 40, 40]
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [192, 192, 96, 3]],  # 12: 11->12, RepNCSPELAN4, torch.Size([b, 192, 40, 40]
   # up-concat merge
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],# 13: Upsample, 12->13, torch.Size([b, 192, 80, 80]
   [[-1, 4], 1, Concat, [1]],  # 14: cat backbone P3, [13, 4], torch.Size([b, 320, 80, 80])
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [128, 128, 64, 3]], # 15: 14->15, RepNCSPELAN4, torch.Size([b, 128, 80, 80]
   # avg-conv-down merge
   [-1, 1, AConv, [96]], # 16: 15->16, AvgPool(2, 1)+CBS, conv(128, 96, 3, 2), torch.Size([b, 96, 40, 40])
   [[-1, 12], 1, Concat, [1]],  # 17: cat head P4, [16, 12], torch.Size([b, 288, 40, 40])
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [192, 192, 96, 3]],  # 18: 17->18, (P4/16-medium), RepNCSPELAN4, torch.Size([b, 192, 40, 40]
   # avg-conv-down merge
   [-1, 1, AConv, [128]], # 19: 18->19, AvgPool(2, 1)+CBS, conv(192, 128, 3, 2), torch.Size([b, 128, 20, 20])
   [[-1, 9], 1, Concat, [1]],  # 20: cat head P5, [19, 9], torch.Size([b, 384, 20, 20])
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [256, 256, 128, 3]],  # 21: 20->21, (P5/32-large), RepNCSPELAN4, torch.Size([b, 256, 20, 20]
   # elan-spp block
   [8, 1, SPPELAN, [256, 128]],  # 22: 8->22, SPPELAN, torch.Size([b, 256, 20, 20])
   # up-concat merge
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 23: 22->23, Upsample, torch.Size([b, 256, 40, 40]
   [[-1, 6], 1, Concat, [1]],  # 24: cat backbone P4, [23, 6], torch.Size([b, 448, 40, 40]
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [192, 192, 96, 3]],  # 25: 24->25, RepNCSPELAN4, torch.Size([b, 192, 40, 40]
   # up-concat merge
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],# 26: 25->26, Upsample, torch.Size([b, 192, 80, 80]
   [[-1, 4], 1, Concat, [1]],  # 27: cat backbone P3, [26, 4], torch.Size([b, 320, 80, 80])
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [128, 128, 64, 3]],  # 28: 27->28, RepNCSPELAN4, torch.Size([b, 128, 80, 80]

   # detect
   [torch.Size([b, 128, 80, 80]), torch.Size([b, 192, 40, 40]), torch.Size([b, 256, 20, 20]), torch.Size([b, 128, 80, 80]), torch.Size([b, 192, 40, 40]), torch.Size([b, 256, 20, 20])]
   [[28, 25, 22, 15, 18, 21], 1, DualDDetect, [nc]],  # Detect(P3, P4, P5)，参考YOLOv8
   # d1: 多尺度辅助信息, torch.Size([b, 144, 80, 80]), torch.Size([b, 144, 40, 40]), torch.Size([b, 144, 20, 20])
   # d2: 主干分支, torch.Size([b, 144, 80, 80]), torch.Size([b, 144, 40, 40]), torch.Size([b, 144, 20, 20])
  ]


PGI
# YOLOv9 backbone
backbone:
  [
   [-1, 1, Silence, []],  # 0: input->0, torch.Size([b, 3, 640, 640])
   # conv down
   [-1, 1, Conv, [64, 3, 2]],  # 1: 0->1, P1/2 CBS, conv(3, 64, 3, 2), torch.Size([b, 64, 320, 320])
   # conv down
   [-1, 1, Conv, [128, 3, 2]], # 2: 1->2, P2/4 CBS, conv(64, 128, 3, 2), torch.Size([b, 128, 160, 160])
   # elan-1 block
   [-1, 1, RepNCSPELAN4, [256, 128, 64, 1]],  # 3: 2->3, CSP+ELAN, torch.Size([b, 256, 160, 160])
   # avg-conv down
   [-1, 1, ADown, [256]],  # 4: 3->4, P3/8, AvgPool(2, 1)+CBS, torch.Size([b, 256, 80, 80])
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [512, 256, 128, 1]],  # 5: 4->5, CSP+ELAN, torch.Size([b, 512, 80, 80]) 
   # avg-conv down
   [-1, 1, ADown, [512]],  # 6: 5->6, P4/16, AvgPool(2, 1)+CBS, torch.Size([b, 512, 40, 40])
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 7: 6->7, CSP+ELAN, torch.Size([b, 512, 40, 40]) 
   # avg-conv down
   [-1, 1, ADown, [512]],  # 8: 7->8, P5/32, AvgPool(2, 1)+CBS, torch.Size([b, 512, 20, 20])
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 9: 8->9, CSP+ELAN, torch.Size([b, 512, 20, 20])
  ]
# YOLOv9 head
head:
  [
   # elan-spp block
   [-1, 1, SPPELAN, [512, 256]],  # 10: 9->10, SPPELAN, torch.Size([b, 512, 20, 20])
   # up-concat merge
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],# 11: 10->11, Upsample, torch.Size([b, 512, 40, 40]
   [[-1, 7], 1, Concat, [1]],  # 12: cat backbone P4, [11, 7], torch.Size([b, 1024, 40, 40]
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 13: 12->13, CSP+ELAN, torch.Size([b, 512, 40, 40]
   # up-concat merge
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 14: Upsample, 13->14, torch.Size([b, 512, 80, 80]
   [[-1, 5], 1, Concat, [1]],  # 15: cat backbone P3, [14, 5], torch.Size([b, 1024, 80, 80])
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [256, 256, 128, 1]],  # 16: 15->16, (P3/8-small): CSP+ELAN, torch.Size([b, 256, 80, 80]
   # avg-conv-down merge
   [-1, 1, ADown, [256]], # 17: 16->17, AvgPool(2, 1)+CBS, torch.Size([b, 256, 40, 40])
   [[-1, 13], 1, Concat, [1]],  # 18: cat head P4, [17, 13], torch.Size([b, 768, 40, 40])
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 19: 18->19, (P4/16-medium), CSP+ELAN, torch.Size([b, 512, 40, 40]
   # avg-conv-down merge
   [-1, 1, ADown, [512]], # 20: 19->20, AvgPool(2, 1)+CBS, torch.Size([b, 512, 20, 20])
   [[-1, 10], 1, Concat, [1]],  # 21: cat head P5, [20, 10], torch.Size([b, 1024, 20, 20])
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 22: 21->22, (P5/32-large), CSP+ELAN, torch.Size([b, 512, 20, 20]
   
   # multi-level reversible auxiliary branch
   # routing
   [5, 1, CBLinear, [[256]]], # 23: 5->23, Conv+Split, [torch.Size([b, 256, 80, 80])]
   [7, 1, CBLinear, [[256, 512]]], # 24: 7->24, Conv+Split, [torch.Size([b, 256, 40, 40]), torch.Size([b, 512, 40, 40])]
   [9, 1, CBLinear, [[256, 512, 512]]], # 25: 9->25, Conv+Split, [torch.Size([b, 256, 20, 20]), torch.Size([b, 512, 20, 20]), torch.Size([b, 512, 20, 20])]
   # conv down
   [0, 1, Conv, [64, 3, 2]],  # 26: 0->26, P1/2, CBS, conv(3, 64, 3, 2), torch.Size([b, 64, 320, 320])
   # conv down
   [-1, 1, Conv, [128, 3, 2]],# 27: 26->27, P2/4, CBS, conv(64, 128, 3, 2), torch.Size([b, 128, 160, 160])
   # elan-1 block
   [-1, 1, RepNCSPELAN4, [256, 128, 64, 1]],  # 28: 27->28, CSP+ELAN, torch.Size([b, 256, 160, 160])
   # avg-conv down fuse
   [-1, 1, ADown, [256]],  # 29: 28->29, P3/8, AvgPool(2, 1)+CBS, torch.Size([b, 256, 80, 80])
   [[23, 24, 25, -1], 1, CBFuse, [[0, 0, 0]]], # 30: [23-0, 24-0, 25-0, 29], Upsample+Sum, torch.Size([b, 256, 80, 80])
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [512, 256, 128, 1]],  # 31: 30->31, CSP+ELAN, torch.Size([b, 512, 80, 80])
   # avg-conv down fuse
   [-1, 1, ADown, [512]],  # 32: 31->32, P4/16, AvgPool(2, 1)+CBS, torch.Size([b, 512, 40, 40])
   [[24, 25, -1], 1, CBFuse, [[1, 1]]], # 33: [24-1, 24-1, 32], Upsample+Sum, torch.Size([b, 512, 40, 40])
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 34: 33->34, CSP+ELAN, torch.Size([b, 512, 40, 40])
   # avg-conv down fuse
   [-1, 1, ADown, [512]],  # 35: 34->35, P5/32, AvgPool(2, 1)+CBS, torch.Size([b, 512, 20, 20])
   [[25, -1], 1, CBFuse, [[2]]], # 36: [25-2, 34], Upsample+Sum, torch.Size([b, 512, 20, 20])
   # elan-2 block
   [-1, 1, RepNCSPELAN4, [512, 512, 256, 1]],  # 37: 36->37, CSP+ELAN, torch.Size([b, 512, 20, 20])
  ]
