DeepLabV2(
  (layer1): _Stem(
    (conv1): _ConvBnReLU(
      (conv): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn): BatchNorm2d(64, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      (relu): ReLU()
    )
    (pool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=True)
  )
  (layer2): _ResLayer(
    (block1): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): _ConvBnReLU(
        (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
    )
    (block2): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block3): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(64, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
  )
  (layer3): _ResLayer(
    (block1): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): _ConvBnReLU(
        (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
    )
    (block2): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block3): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block4): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(128, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
  )
  (layer4): _ResLayer(
    (block1): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): _ConvBnReLU(
        (conv): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
    )
    (block2): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block3): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block4): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block5): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block6): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block7): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block8): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block9): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block10): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block11): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block12): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block13): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block14): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block15): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block16): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block17): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block18): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block19): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block20): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block21): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block22): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block23): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
        (bn): BatchNorm2d(256, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(1024, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
  )
  (layer5): _ResLayer(
    (block1): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): _ConvBnReLU(
        (conv): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
    )
    (block2): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
    (block3): _Bottleneck(
      (reduce): _ConvBnReLU(
        (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (conv3x3): _ConvBnReLU(
        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), bias=False)
        (bn): BatchNorm2d(512, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
        (relu): ReLU()
      )
      (increase): _ConvBnReLU(
        (conv): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(2048, eps=1e-05, momentum=0.0010000000000000009, affine=True, track_running_stats=True)
      )
      (shortcut): Identity()
    )
  )
(aspp): _ASPP(
  (c0): Conv2d(2048, 21, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6))
  (c1): Conv2d(2048, 21, kernel_size=(3, 3), stride=(1, 1), padding=(12, 12), dilation=(12, 12))
  (c2): Conv2d(2048, 21, kernel_size=(3, 3), stride=(1, 1), padding=(18, 18), dilation=(18, 18))
  (c3): Conv2d(2048, 21, kernel_size=(3, 3), stride=(1, 1), padding=(24, 24), dilation=(24, 24))
)
)