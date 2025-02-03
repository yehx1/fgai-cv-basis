wget https://ghfast.top/https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

data/coco.yaml

train: /root/project/fgai-cv-basis/01_data/coco128/images
val: /root/project/fgai-cv-basis/01_data/coco128/images
test: /root/project/fgai-cv-basis/01_data/coco128/images
# number of classes
nc: 80
# class names
names: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush' ]


parser.add_argument('--weights', type=str, default='yolov7.pt', help='initial weights path')
parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')


anchors = torch.cat(anchors) # torch.Size([8400, 4])，anchor box坐标，xyxy
anchor_points = torch.cat(anchor_points).to(device) # torch.Size([8400, 2]), 特征网格中心在原图坐标
stride_tensor = torch.cat(stride_tensor).to(device) # torch.Size([8400, 1]), 每个锚点对应的缩放尺度stride
# return anchors, anchor_points, num_anchors_list, stride_tensor


Sequential(
  (0): Conv(
    (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (1): Conv(
    (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (2): Conv(
    (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (3): Conv(
    (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (4): Conv(
    (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (5): Conv(
    (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (6): Conv(
    (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (7): Conv(
    (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (8): Conv(
    (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (9): Conv(
    (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (10): Concat()
  (11): Conv(
    (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (12): MP(
    (m): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (13): Conv(
    (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (14): Conv(
    (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (15): Conv(
    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (16): Concat()
  (17): Conv(
    (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (18): Conv(
    (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (19): Conv(
    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (20): Conv(
    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (21): Conv(
    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (22): Conv(
    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (23): Concat()
  (24): Conv(
    (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (25): MP(
    (m): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (26): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (27): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (28): Conv(
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (29): Concat()
  (30): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (31): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (32): Conv(
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (33): Conv(
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (34): Conv(
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (35): Conv(
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (36): Concat()
  (37): Conv(
    (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(1024, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (38): MP(
    (m): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (39): Conv(
    (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (40): Conv(
    (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (41): Conv(
    (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (42): Concat()
  (43): Conv(
    (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (44): Conv(
    (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (45): Conv(
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (46): Conv(
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (47): Conv(
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (48): Conv(
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (49): Concat()
  (50): Conv(
    (conv): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(1024, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (51): SPPCSPC(
    (cv1): Conv(
      (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (cv3): Conv(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (cv4): Conv(
      (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (m): ModuleList(
      (0): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
      (1): MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
      (2): MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)
    )
    (cv5): Conv(
      (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (cv6): Conv(
      (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (cv7): Conv(
      (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
      (act): SiLU()
    )
  )
  (52): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (53): Upsample(scale_factor=2.0, mode='nearest')
  (54): Conv(
    (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (55): Concat()
  (56): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (57): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (58): Conv(
    (conv): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (59): Conv(
    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (60): Conv(
    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (61): Conv(
    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (62): Concat()
  (63): Conv(
    (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (64): Conv(
    (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (65): Upsample(scale_factor=2.0, mode='nearest')
  (66): Conv(
    (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (67): Concat()
  (68): Conv(
    (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (69): Conv(
    (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (70): Conv(
    (conv): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (71): Conv(
    (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (72): Conv(
    (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (73): Conv(
    (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (74): Concat()
  (75): Conv(
    (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (76): MP(
    (m): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (77): Conv(
    (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (78): Conv(
    (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (79): Conv(
    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (80): Concat()
  (81): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (82): Conv(
    (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (83): Conv(
    (conv): Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (84): Conv(
    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (85): Conv(
    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (86): Conv(
    (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (87): Concat()
  (88): Conv(
    (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (89): MP(
    (m): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (90): Conv(
    (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (91): Conv(
    (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (92): Conv(
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (93): Concat()
  (94): Conv(
    (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (95): Conv(
    (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (96): Conv(
    (conv): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (97): Conv(
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (98): Conv(
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (99): Conv(
    (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (bn): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (100): Concat()
  (101): Conv(
    (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (bn): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (102): RepConv(
    (act): SiLU()
    (rbr_dense): Sequential(
      (0): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    )
    (rbr_1x1): Sequential(
      (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    )
  )
  (103): RepConv(
    (act): SiLU()
    (rbr_dense): Sequential(
      (0): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    )
    (rbr_1x1): Sequential(
      (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    )
  )
  (104): RepConv(
    (act): SiLU()
    (rbr_dense): Sequential(
      (0): Conv2d(512, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (1): BatchNorm2d(1024, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    )
    (rbr_1x1): Sequential(
      (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(1024, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
    )
  )
  (105): Detect(
    (m): ModuleList(
      (0): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
      (1): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
      (2): Conv2d(1024, 255, kernel_size=(1, 1), stride=(1, 1))
    )
  )
)

[3, 4, 5, 7, 11, 13, 16, 17, 18, 20, 24, 24, 26, 29, 30, 31, 33, 37, 37, 39, 42, 43, 44, 46, 51, 53, 55, 56, 57, 58, 59, 60, 63, 65, 67, 68, 69, 70, 71, 72, 75, 75, 77, 80, 81, 82, 83, 84, 85, 88, 88, 90, 93, 94, 95, 96, 97, 98, 101, ...]
torch.Size([b, 3, 640, 640])
https://zhuanlan.zhihu.com/p/649417999
https://blog.csdn.net/weixin_52862386/article/details/139941871
https://docs.ultralytics.com/zh/models/yolov7/#citations-and-acknowledgements

0: CBS, conv(3, 32, 3, 1), torch.Size([b, 32, 640, 640])
1: CBS, conv(32, 64, 3, 2), torch.Size([b, 64, 320, 320])
2: CBS, conv(64, 64, 3, 1), torch.Size([b, 64, 320, 320])
3: CBS, conv(64, 128, 3, 2), torch.Size([b, 128, 160, 160])
4: CBS, conv(128, 64, 1, 1), torch.Size([b, 64, 160, 160])
5: CBS, conv(128, 64, 1, 1), torch.Size([b, 64, 160, 160])
6: CBS, conv(64, 64, 3, 1), torch.Size([b, 64, 160, 160])
7: CBS, conv(64, 64, 3, 1), torch.Size([b, 64, 160, 160])
8: CBS, conv(64, 64, 3, 1), torch.Size([b, 64, 160, 160])
9: CBS, conv(64, 64, 3, 1), torch.Size([b, 64, 160, 160])
10: [9, 7, 5, 4], concate, torch.Size([b, 256, 160, 160])
11: CBS, conv(256, 256, 1, 1), torch.Size([b, 256, 160, 160])
12: MP, MaxPool2s(2, 2), torch.Size([b, 256, 80, 80])
13: 12->13, CBS, conv(256, 128, 1, 1), torch.Size([b, 128, 80, 80])
14: 11->14, CBS, conv(256, 128, 1, 1), torch.Size([b, 128, 160, 160])
15: 14->15, CBS, conv(128, 128, 3, 2), torch.Size([b, 128, 80, 80])
16: [15, 13], concate, torch.Size([b, 256, 80, 80])

# yolov7 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Conv, [32, 3, 1]],  # 0: input->0: CBS, conv(3, 32, 3, 1), torch.Size([b, 32, 640, 640])
   [-1, 1, Conv, [64, 3, 2]],  # 1: 0->1: 1-P1/2, CBS, conv(32, 64, 3, 2), torch.Size([b, 64, 320, 320])
   [-1, 1, Conv, [64, 3, 1]],  # 2：1->2: CBS, conv(64, 64, 3, 1), torch.Size([b, 64, 320, 320])
   [-1, 1, Conv, [128, 3, 2]], # 3: 2->3: 3-P2/4, CBS, conv(64, 128, 3, 2), torch.Size([b, 128, 160, 160])
   # ELAN1
   [-1, 1, Conv, [64, 1, 1]],  # 4: 3->4, CBS, conv(128, 64, 1, 1), torch.Size([b, 64, 160, 160])     
   [-2, 1, Conv, [64, 1, 1]],  # 5: 3->5, CBS, conv(128, 64, 1, 1), torch.Size([b, 64, 160, 160]) 
   [-1, 1, Conv, [64, 3, 1]],  # 6: 5->6, CBS, conv(64, 64, 3, 1), torch.Size([b, 64, 160, 160]) 
   [-1, 1, Conv, [64, 3, 1]],  # 7: 6->7, CBS, conv(64, 64, 3, 1), torch.Size([b, 64, 160, 160]) 
   [-1, 1, Conv, [64, 3, 1]],  # 8: 7->8, CBS, conv(64, 64, 3, 1), torch.Size([b, 64, 160, 160]) 
   [-1, 1, Conv, [64, 3, 1]],  # 9: 8->9, CBS, conv(64, 64, 3, 1), torch.Size([b, 64, 160, 160]) 
   [[-1, -3, -5, -6], 1, Concat, [1]], # 10: [9, 7, 5, 4], concate, torch.Size([b, 256, 160, 160])
   [-1, 1, Conv, [256, 1, 1]],  # 11: 10->11, CBS, conv(256, 256, 1, 1), torch.Size([b, 256, 160, 160])
   # ELAN1 end
   # MPC1
   [-1, 1, MP, []],             # 12: 11->12, MP, MaxPool2s(2, 2), torch.Size([b, 256, 80, 80])
   [-1, 1, Conv, [128, 1, 1]],  # 13: 12->13, CBS, conv(256, 128, 1, 1), torch.Size([b, 128, 80, 80])
   [-3, 1, Conv, [128, 1, 1]],  # 14: 11->14, CBS, conv(256, 128, 1, 1), torch.Size([b, 128, 160, 160])
   [-1, 1, Conv, [128, 3, 2]],  # 15: 14->15, CBS, conv(128, 128, 3, 2), torch.Size([b, 128, 80, 80])
   [[-1, -3], 1, Concat, [1]],  # 16-P3/8, [15, 13], concate, torch.Size([b, 256, 80, 80])
   # MPC1 end
   # ELAN2
   [-1, 1, Conv, [128, 1, 1]],  # 17
   [-2, 1, Conv, [128, 1, 1]],  # 18
   [-1, 1, Conv, [128, 3, 1]],  # 19
   [-1, 1, Conv, [128, 3, 1]],  # 20
   [-1, 1, Conv, [128, 3, 1]],  # 21
   [-1, 1, Conv, [128, 3, 1]],  # 22
   [[-1, -3, -5, -6], 1, Concat, [1]],  # 23, torch.Size([b, 512, 80, 80])
   [-1, 1, Conv, [512, 1, 1]],  # 24
   # ELAN2 End
   # MPC2
   [-1, 1, MP, []],             # 25
   [-1, 1, Conv, [256, 1, 1]],  # 26
   [-3, 1, Conv, [256, 1, 1]],  # 27
   [-1, 1, Conv, [256, 3, 2]],  # 28
   [[-1, -3], 1, Concat, [1]],  # 29-P4/16, torch.Size([b, 512, 40, 40])
   # MPC2 end
   # ELAN3
   [-1, 1, Conv, [256, 1, 1]],  # 30
   [-2, 1, Conv, [256, 1, 1]],  # 31
   [-1, 1, Conv, [256, 3, 1]],  # 32
   [-1, 1, Conv, [256, 3, 1]],  # 33
   [-1, 1, Conv, [256, 3, 1]],  # 34
   [-1, 1, Conv, [256, 3, 1]],  # 35
   [[-1, -3, -5, -6], 1, Concat, [1]], # 36, torch.Size([b, 1024, 40, 40])
   [-1, 1, Conv, [1024, 1, 1]], # 37
   # ELAN3 End
   # MPC3
   [-1, 1, MP, []],             # 38
   [-1, 1, Conv, [512, 1, 1]],  # 39
   [-3, 1, Conv, [512, 1, 1]],  # 40
   [-1, 1, Conv, [512, 3, 2]],  # 41
   [[-1, -3], 1, Concat, [1]],  # 42-P5/32, torch.Size([b, 1024, 20, 20])
   # MPC3 end
   # ELAN4
   [-1, 1, Conv, [256, 1, 1]],  # 43
   [-2, 1, Conv, [256, 1, 1]],  # 44
   [-1, 1, Conv, [256, 3, 1]],  # 45
   [-1, 1, Conv, [256, 3, 1]],  # 46
   [-1, 1, Conv, [256, 3, 1]],  # 47
   [-1, 1, Conv, [256, 3, 1]],  # 48
   [[-1, -3, -5, -6], 1, Concat, [1]],  # 49
   [-1, 1, Conv, [1024, 1, 1]],  # 50, torch.Size([b, 1024, 20, 20])
   # ELAN4 End
  ]

# yolov7 head
head:
  [[-1, 1, SPPCSPC, [512]],    # 51, 50->51, SPP+CSP+CBS, torch.Size([b, 512, 20, 20])
   # Feature fuse 1
   [-1, 1, Conv, [256, 1, 1]], # 52: 51->52, CBS, conv(512, 256, 1, 1), torch.Size([b, 256, 20, 20])
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 53: 52->53, Upsample, torch.Size([b, 256, 40, 40])
   [37, 1, Conv, [256, 1, 1]], # 54: 37->54, route backbone P4, , CBS, conv(1024, 256, 1, 1), torch.Size([b, 256, 40, 240])
   [[-1, -2], 1, Concat, [1]], # 55: [53, 54], concate, torch.Size([b, 512, 40, 40])
   # Feature fuse end
   # ELAN-W 1
   [-1, 1, Conv, [256, 1, 1]], # 56: 55->56, CBS, conv(512, 256, 1, 1), torch.Size([b, 256, 40, 40])
   [-2, 1, Conv, [256, 1, 1]], # 57: 55->57, CBS, conv(512, 256, 1, 1), torch.Size([b, 256, 40, 40])
   [-1, 1, Conv, [128, 3, 1]], # 58: 57->58, CBS, conv(256, 128, 3, 1), torch.Size([b, 128, 40, 40])
   [-1, 1, Conv, [128, 3, 1]], # 59: 58->59, CBS, conv(128, 128, 3, 1), torch.Size([b, 128, 40, 40])
   [-1, 1, Conv, [128, 3, 1]], # 60: 59->60, CBS, conv(128, 128, 3, 1), torch.Size([b, 128, 40, 40])
   [-1, 1, Conv, [128, 3, 1]], # 61: 60->61, CBS, conv(128, 128, 3, 1), torch.Size([b, 128, 40, 40])
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],# 62: [61, 60, 59, 58, 57, 56], concate, torch.Size([b, 1024, 40, 40])
   [-1, 1, Conv, [256, 1, 1]], # 63: 60->61, CBS, conv(1024, 256, 1, 1), torch.Size([b, 256, 40, 40])
   # ELAN-W 1 end
   # Feature fuse 2
   [-1, 1, Conv, [128, 1, 1]], # 64
   [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 65
   [24, 1, Conv, [128, 1, 1]], # route backbone P3 # 66
   [[-1, -2], 1, Concat, [1]], # 67, torch.Size([b, 256, 80, 80])
   # Feature fuse 2 end
   # ELAN-W 2
   [-1, 1, Conv, [128, 1, 1]],
   [-2, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [-1, 1, Conv, [64, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [128, 1, 1]], # 75, torch.Size([b, 128, 80, 80])
   # ELAN-W 2 end
   # MPC4
   [-1, 1, MP, []],
   [-1, 1, Conv, [128, 1, 1]],
   [-3, 1, Conv, [128, 1, 1]],
   [-1, 1, Conv, [128, 3, 2]],
   [[-1, -3, 63], 1, Concat, [1]], # 80, torch.Size([b, 256, 40, 40])
   # MPC4 end
   # ELAN-W 3
   [-1, 1, Conv, [256, 1, 1]],
   [-2, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [-1, 1, Conv, [128, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [256, 1, 1]], # 88, torch.Size([b, 256, 40, 40])
   # ELAN-W 3 end
   # MPC5
   [-1, 1, MP, []],
   [-1, 1, Conv, [256, 1, 1]],
   [-3, 1, Conv, [256, 1, 1]],
   [-1, 1, Conv, [256, 3, 2]],
   [[-1, -3, 51], 1, Concat, [1]], # 93, torch.Size([b, 1024, 40, 40])
   # MPC5 end
   # ELAN-W 4
   [-1, 1, Conv, [512, 1, 1]],
   [-2, 1, Conv, [512, 1, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [-1, 1, Conv, [256, 3, 1]],
   [[-1, -2, -3, -4, -5, -6], 1, Concat, [1]],
   [-1, 1, Conv, [512, 1, 1]], # 101, torch.Size([b, 512, 20, 20])
   # ELAN-W 4 end
   [75, 1, RepConv, [256, 3, 1]],   # 102: 75->102, RepConv, torch.Size([b, 256, 80, 80])
   [88, 1, RepConv, [512, 3, 1]],   # 103: 88->103, RepConv, torch.Size([b, 512, 40, 40])
   [101, 1, RepConv, [1024, 3, 1]], # 104: 101->104, RepConv, torch.Size([b, 1024, 20, 20])

   [[102,103,104], 1, Detect, [nc, anchors]], # 105, Detect(P3, P4, P5)
   # 105, out, [torch.Size([b, 3, 80, 80, 85]), torch.Size([b, 3, 40, 40, 85]), torch.Size([b, 3, 20, 20, 85])]
  ]
4.0 1.0, 0.4
lcls, BCE Loss, 0.3
lbox, 0.05, iou
lobj, BCE Loss, 0.7

# find_3_positive、find_5_positive
def find_3_positive(self, p, targets):
  # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
  na, nt = self.na, targets.shape[0]  # number of anchors, targets
  indices, anch = [], []
  gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
  ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
  targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
  g = 0.5  # bias
  off = torch.tensor([[0, 0],
                      [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                      # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                      ], device=targets.device).float() * g  # offsets
  for i in range(self.nl):
      anchors = self.anchors[i]
      gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
      # Match targets to anchors
      t = targets * gain
      if nt:
          # Matches
          r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
          j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
          # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
          t = t[j]  # filter
          # Offsets
          gxy = t[:, 2:4]  # grid xy
          gxi = gain[[2, 3]] - gxy  # inverse
          j, k = ((gxy % 1. < g) & (gxy > 1.)).T
          l, m = ((gxi % 1. < g) & (gxi > 1.)).T
          j = torch.stack((torch.ones_like(j), j, k, l, m))
          t = t.repeat((5, 1, 1))[j]
          offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
      else:
          t = targets[0]
          offsets = 0
      # Define
      b, c = t[:, :2].long().T  # image, class
      gxy = t[:, 2:4]  # grid xy
      gwh = t[:, 4:6]  # grid wh
      gij = (gxy - offsets).long()
      gi, gj = gij.T  # grid xy indices
      # Append
      a = t[:, 6].long()  # anchor indices
      indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
      anch.append(anchors[a])  # anchors
  return indices, anch

def find_5_positive(self, p, targets):
  # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
  na, nt = self.na, targets.shape[0]  # number of anchors, targets
  indices, anch = [], []
  gain = torch.ones(7, device=targets.device).long()  # normalized to gridspace gain
  ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
  targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
  g = 1.0  # bias
  off = torch.tensor([[0, 0],
                      [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                      # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                      ], device=targets.device).float() * g  # offsets
  for i in range(self.nl):
      anchors = self.anchors[i]
      gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
      # Match targets to anchors
      t = targets * gain
      if nt:
          # Matches
          r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
          j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
          # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
          t = t[j]  # filter
          # Offsets
          gxy = t[:, 2:4]  # grid xy
          gxi = gain[[2, 3]] - gxy  # inverse
          j, k = ((gxy % 1. < g) & (gxy > 1.)).T
          l, m = ((gxi % 1. < g) & (gxi > 1.)).T
          j = torch.stack((torch.ones_like(j), j, k, l, m))
          t = t.repeat((5, 1, 1))[j]
          offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
      else:
          t = targets[0]
          offsets = 0
      # Define
      b, c = t[:, :2].long().T  # image, class
      gxy = t[:, 2:4]  # grid xy
      gwh = t[:, 4:6]  # grid wh
      gij = (gxy - offsets).long()
      gi, gj = gij.T  # grid xy indices
      # Append
      a = t[:, 6].long()  # anchor indices
      indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
      anch.append(anchors[a])  # anchors
  return indices, anch                 



