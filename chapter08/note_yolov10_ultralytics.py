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

# SCDown，使用深度卷积进行下采样
def forward(self, x):
  return self.cv2(self.cv1(x))
SCDown(
  (cv1): Conv(
    (conv): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU(inplace=True)
  )
  (cv2): Conv(
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): Identity()
  )
)

# 8: C2fCIB
def forward(self, x):
  y = list(self.cv1(x).chunk(2, 1))
  y.extend(m(y[-1]) for m in self.m)
  return self.cv2(torch.cat(y, 1))
C2fCIB(
  (cv1): Conv(
    (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU(inplace=True)
  )
  (cv2): Conv(
    (conv): Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU(inplace=True)
  )
  (m): ModuleList(
    (0): CIB(
      (cv1): Sequential(
        (0): Conv(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (1): Conv(
          (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (2): RepVGGDW(
          (conv): Conv(
            (conv): Conv2d(512, 512, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=512, bias=False)
            (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (conv1): Conv(
            (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512, bias=False)
            (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
            (act): Identity()
          )
          (act): SiLU(inplace=True)
        )
        (3): Conv(
          (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
        (4): Conv(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
          (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
          (act): SiLU(inplace=True)
        )
      )
    )
  )
)

# 10: PSA, SPPF->PSA
def forward(self, x):
  a, b = self.cv1(x).split((self.c, self.c), dim=1) # torch.Size([b, 256, 20, 20])
  b = b + self.attn(b) # torch.Size([b, 256, 20, 20])
  b = b + self.ffn(b) # conv 1x1, torch.Size([b, 256, 20, 20])
  return self.cv2(torch.cat((a, b), 1)) #  torch.Size([b, 512, 20, 20]) 

# self.attn
def forward(self, x):
  B, C, H, W = x.shape # b, 256, 20, 20
  N = H * W
  qkv = self.qkv(x) # conv(256, 512, 1, 1), 针对通道，torch.Size([b, 512, 20, 20])
  q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
      [self.key_dim, self.key_dim, self.head_dim], dim=2
  ) # head数量为4，q:torch.Size([b, 4, 32, 400]), k:torch.Size([b, 4, 32, 400]), v:torch.Size([b, 4, 64, 400])
  attn = (q.transpose(-2, -1) @ k) * self.scale # torch.Size([b, 4, 400, 400])
  attn = attn.softmax(dim=-1) # 转为概率权重，torch.Size([b, 4, 400, 400])
  x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W)) # 注意力与空间特征（位置特征）融合
  x = self.proj(x) # 过渡层融合，conv(256, 256, 1, 1)， torch.Size([b, 256, 20, 20])
  return x

# self.ffn
Sequential(
  (0): Conv(
    (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU(inplace=True)
  )
  (1): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): Identity()
  )
)

v10Detect
x_detach = [xi.detach() for xi in x]
one2one = [
  torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
] # [torch.Size([b, 144, 80, 80]), torch.Size([b, 144, 40, 40]), torch.Size([b, 144, 20, 20])]
for i in range(self.nl):
  x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) # [torch.Size([b, 144, 80, 80]), torch.Size([b, 144, 40, 40]), torch.Size([b, 144, 20, 20])]
if self.training:  # Training path
  return {"one2many": x, "one2one": one2one}

# self.cv2[0]，回归头
Sequential(
  (0): Conv(
    (conv): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU(inplace=True)
  )
  (1): Conv(
    (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU(inplace=True)
  )
  (2): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
)

# self.cv3[0]， 分类头
Sequential(
  (0): Sequential(
    (0): Conv(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (1): Conv(
      (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
  )
  (1): Sequential(
    (0): Conv(
      (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
    (1): Conv(
      (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU(inplace=True)
    )
  )
  (2): Conv2d(128, 80, kernel_size=(1, 1), stride=(1, 1))
)

# Loss:
class E2EDetectLoss:
    """Criterion class for computing training losses."""
def __init__(self, model):
  """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
  self.one2many = v8DetectionLoss(model, tal_topk=10)
  self.one2one = v8DetectionLoss(model, tal_topk=1)

def __call__(self, preds, batch):
  """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
  preds = preds[1] if isinstance(preds, tuple) else preds
  one2many = preds["one2many"]
  loss_one2many = self.one2many(one2many, batch)
  one2one = preds["one2one"]
  loss_one2one = self.one2one(one2one, batch)
  return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]



backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [32, 3, 2]]    # 0: input->0, P1/2, CBS, conv(3, 32, 3, 2), torch.Size([b, 32, 320, 320])
  - [-1, 1, Conv, [64, 3, 2]]    # 1: 0->1, P2/4, CBS, conv(32, 64, 3, 2), torch.Size([b, 64, 160, 160])
  - [-1, 1, C2f, [64, True]]     # 2: 1->2, C2f, torch.Size([b, 64, 160, 160])
  - [-1, 1, Conv, [256, 3, 2]]   # 3: 2->3, P3/8, CBS, conv(64, 128, 3, 2), torch.Size([b, 128, 80, 80])
  - [-1, 2, C2f, [128, True]]    # 4: 3->4, C2f2, torch.Size([b, 128, 80, 80]) 
  - [-1, 1, SCDown, [256 3, 2]]  # 5: 4->5, P4/16, SCDown（深度可分离卷积）, torch.Size([b, 256, 40, 40])
  - [-1, 2, C2f, [256, True]]    # 6: 3->4, C2f2, torch.Size([b, 256, 40, 40]) 
  - [-1, 1, SCDown, [512, 3, 2]] # 7: 6->7, P5/32, SCDown（深度可分离卷积）, torch.Size([b, 512, 20, 20])
  - [-1, 1, C2fCIB, [512, True, True]] # 87->8, C2fCIB(C2f+RepVGGDW), torch.Size([b, 512, 20, 20]) 
  - [-1, 1, SPPF, [512, 5]] # 9: 8->9, SPPF, torch.Size([b, 512, 20, 20]) 
  - [-1, 1, PSA, [512]]    # 10: 9->10, PSA, torch.Size([b, 512, 20, 20]) 

# YOLOv10.0n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 11: 10->11, Upsample, torch.Size([b, 512, 40, 40]
  - [[-1, 6], 1, Concat, [1]] # 12:  cat backbone P4 [11, 6], torch.Size([b, 768, 40, 40]
  - [-1, 1, C2f, [512]] # 13: 12->13, C2f, torch.Size([b, 256, 40, 40]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 14: 13->14, Upsample, torch.Size([b, 256, 80, 80]
  - [[-1, 4], 1, Concat, [1]] # 15: cat backbone P3, [14, 4], torch.Size([b, 384, 80, 80])
  - [-1, 3, C2f, [256]] # 16: 15->16, (P3/8-small), C2f, torch.Size([b, 128, 80, 80]

  - [-1, 1, Conv, [256, 3, 2]] # 17: 16->17, conv(128, 128, 3, 2), torch.Size([b, 128, 40, 40])
  - [[-1, 13], 1, Concat, [1]] # 18: cat head P4, [17, 13], torch.Size([b, 384, 40, 40])
  - [-1, 1, C2f, [512]] # 19: 18->19, (P4/16-medium), C2f, torch.Size([b, 256, 40, 40])

  - [-1, 1, SCDown, [512, 3, 2]] # 20: 19->20, SCDown, torch.Size([b, 256, 20, 20]
  - [[-1, 10], 1, Concat, [1]] # 21: cat head P5, [20, 10], torch.Size([b, 768, 20, 20]
  - [-1, 1, C2fCIB, [1024, True, True]] # 22: 21->22, (P5/32-large), C2fCIB(C2f+RepVGGDW), torch.Size([b, 512, 20, 20]) 
  # [torch.Size([b, 128, 80, 80], torch.Size([b, 256, 40, 40]), torch.Size([b, 512, 20, 20])]
  - [[16, 19, 22], 1, v10Detect, [nc]] # Detect(P3, P4, P5)
  # 23
  # one2one: [torch.Size([b, 144, 80, 80]), torch.Size([b, 144, 40, 40]), torch.Size([b, 144, 20, 20])]
  # one2many: [torch.Size([b, 144, 80, 80]), torch.Size([b, 144, 40, 40]), torch.Size([b, 144, 20, 20])]