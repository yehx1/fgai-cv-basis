def forward(self, x: Tensor) -> Dict[str, Tensor]: # torch.Size([b, 3, 256, 256])
    input_shape = x.shape[-2:] # torch.Size([256, 256])
    # contract: features is a dict of tensors
    features = self.backbone(x) # aux: torch.Size([b, 1024, 32, 32]), out: torch.Size([b, 2048, 32, 32])
    result = OrderedDict()
    x = features["out"] # torch.Size([b, 2048, 32, 32])
    x = self.classifier(x) # FCNHead, torch.Size([b, 21, 32, 32])
    x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False) # torch.Size([b, 21, 256, 256])
    result["out"] = x # torch.Size([b, 21, 256, 256])
    return result


self.classifier
FCNHead(
  (0): Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU()
  (3): Dropout(p=0.1, inplace=False)
  (4): Conv2d(512, 21, kernel_size=(1, 1), stride=(1, 1))
)

self.aux_classifier
FCNHead(
  (0): Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU()
  (3): Dropout(p=0.1, inplace=False)
  (4): Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))
)