# 训练入口
line 380
/root/miniconda3/envs/cv/lib/python3.12/site-packages/ultralytics/engine/trainer.py

# 推理入口
line 145
/root/miniconda3/envs/cv/lib/python3.12/site-packages/ultralytics/nn/tasks.py

# 损失计算入口
line 739
/root/miniconda3/envs/cv/lib/python3.12/site-packages/ultralytics/utils/loss.py

# 配置文件
/root/miniconda3/envs/cv/lib/python3.12/site-packages/ultralytics/cfg/models/v10/yolov10s.yaml

# YOLOv8 C2f
def forward(self, x):
  """Forward pass through C2f layer."""
  y = list(self.cv1(x).chunk(2, 1))
  y.extend(m(y[-1]) for m in self.m)
  return self.cv2(torch.cat(y, 1))

C2f(
  (cv1): Conv(
    (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(96, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (m): ModuleList(
    (0): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): SiLU()
      )
      (cv2): Conv(
        (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): SiLU()
      )
    )
  )
)

C3k2 False C2f
C3k2(
  (cv1): Conv(
    (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU(inplace=True)
  )
  (cv2): Conv(
    (conv): Conv2d(96, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU(inplace=True)
  )
  (m): ModuleList(
    (0): Bottleneck(
      (cv1): Conv(
        (conv): Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(16, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
    )
  )
)

C3k2 True C2f
C3k2(
  (cv1): Conv(
    (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act): SiLU(inplace=True)
  )
  (cv2): Conv(
    (conv): Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act): SiLU(inplace=True)
  )
  (m): ModuleList(
    (0): C3k(
      (cv1): Conv(
        (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv2): Conv(
        (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (cv3): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (act): SiLU(inplace=True)
      )
      (m): Sequential(
        (0): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
        (1): Bottleneck(
          (cv1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
          (cv2): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (act): SiLU(inplace=True)
          )
        )
      )
    )
  )
)

C2PSA
def forward(self, x):
    """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
    a, b = self.cv1(x).split((self.c, self.c), dim=1)
    b = self.m(b)
    return self.cv2(torch.cat((a, b), 1))

C2PSA(
  (cv1): Conv(
    (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (cv2): Conv(
    (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (m): Sequential(
    (0): PSABlock(
      (attn): Attention(
        (qkv): Conv(
          (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): Identity()
        )
        (proj): Conv(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): Identity()
        )
        (pe): Conv(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): Identity()
        )
      )
      (ffn): Sequential(
        (0): Conv(
          (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (1): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): Identity()
        )
      )
    )
  )
)




nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024] # summary: 319 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
  s: [0.50, 0.50, 1024] # summary: 319 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
  m: [0.50, 1.00, 512] # summary: 409 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
  l: [1.00, 1.00, 512] # summary: 631 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
  x: [1.00, 1.50, 512] # summary: 631 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

# YOLO11s backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [32, 3, 2]]  # 0: input->0, P1/2, CBS, conv(3, 32, 3, 2), torch.Size([b, 32, 320, 320])
  - [-1, 1, Conv, [64, 3, 2]]  # 1: 0->1, P2/4, CBS, conv(32, 64, 3, 2), torch.Size([b, 64, 160, 160])
  - [-1, 2, C3k2, [128, False, 0.25]]# 2: 1->2, C3k2 False, torch.Size([b, 128, 160, 160])
  - [-1, 1, Conv, [128, 3, 2]] # 3: 2->3, P3/8, CBS, conv(128, 128, 3, 2), torch.Size([b, 128, 80, 80])
  - [-1, 2, C3k2, [256, False, 0.25]]# 4: 3->4, C3k2 False, torch.Size([b, 256, 80, 80]) 
  - [-1, 1, Conv, [256, 3, 2]] # 5: 4->5, P4/16, CBS, conv(256, 256, 3, 2), torch.Size([b, 256, 40, 40])
  - [-1, 2, C3k2, [512, True]] # 6: 5->6, C3k2 True, torch.Size([b, 256, 40, 40]) 
  - [-1, 1, Conv, [512, 3, 2]] # 7: 6->7, P5/32, CBS, conv(256, 512, 3, 2), torch.Size([b, 512, 20, 20])
  - [-1, 2, C3k2, [512, True]] # 87->8, C3k2 True, torch.Size([b, 512, 20, 20]) 
  - [-1, 1, SPPF, [512, 5]] # 9: 8->9, SPPF, torch.Size([b, 512, 20, 20]) 
  - [-1, 2, C2PSA, [512]]  # 10: 9->10, C2PSA, torch.Size([b, 512, 20, 20])

# YOLO11n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]# 11: 10->11, Upsample, torch.Size([b, 512, 40, 40]
  - [[-1, 6], 1, Concat, [1]]   # 12:  cat backbone P4 [11, 6], torch.Size([b, 768, 40, 40]
  - [-1, 2, C3k2, [256, False]] # 13: 12->13, C3k2 False, torch.Size([b, 256, 40, 40]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]# 14: 13->14, Upsample, torch.Size([b, 256, 80, 80]
  - [[-1, 4], 1, Concat, [1]]   # 15: cat backbone P3, [14, 4], torch.Size([b, 512, 80, 80])
  - [-1, 2, C3k2, [128, False]] # 16: 15->16, (P3/8-small), C3k2 False, torch.Size([b, 128, 80, 80]

  - [-1, 1, Conv, [128, 3, 2]]  # 17: 16->17, CBS, conv(128, 128, 3, 2), torch.Size([b, 128, 40, 40])
  - [[-1, 13], 1, Concat, [1]]  # 18: cat head P4, [17, 13], torch.Size([b, 384, 40, 40])
  - [-1, 2, C3k2, [256, False]] # 19: 18->19, (P4/16-medium), C3k2 False, torch.Size([b, 256, 40, 40])

  - [-1, 1, Conv, [256, 3, 2]] # 20: 19->20, CBS, conv(256, 256, 3, 2), torch.Size([b, 256, 20, 20]
  - [[-1, 10], 1, Concat, [1]] # 21: cat head P5, [20, 10], torch.Size([b, 768, 20, 20]
  - [-1, 2, C3k2, [512, True]]# 22: 21->22, (P5/32-large), C3k2 True, torch.Size([b, 512, 20, 20]) 
  # [torch.Size([b, 128, 80, 80], torch.Size([b, 256, 40, 40]), torch.Size([b, 512, 20, 20])]
  - [[16, 19, 22], 1, Detect, [nc]] # Detect(P3, P4, P5)
  # 23: [torch.Size([b, 144, 80, 80]), torch.Size([b, 144, 40, 40]), torch.Size([b, 144, 20, 20])]
  # v8DetectionLoss






