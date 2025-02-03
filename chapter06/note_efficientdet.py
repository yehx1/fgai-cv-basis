def _forward_fast_attention(self, inputs):
    if self.first_time:
        p3, p4, p5 = inputs
        # torch.Size([b, 48, 96, 96]), torch.Size([b, 120, 48, 48]), torch.Size([b, 352, 24, 24])
        p6_in = self.p5_to_p6(p5) # torch.Size([b, 112, 12, 12])
        p7_in = self.p6_to_p7(p6_in) # torch.Size([b, 112, 6, 6])
        p3_in = self.p3_down_channel(p3) # torch.Size([b, 112, 96, 96])
        p4_in = self.p4_down_channel(p4) # torch.Size([b, 112, 48, 48])
        p5_in = self.p5_down_channel(p5) # torch.Size([b, 112, 24, 24])

    else:
        # P3_0, P4_0, P5_0, P6_0 and P7_0
        p3_in, p4_in, p5_in, p6_in, p7_in = inputs

    # P7_0 to P7_2
    # Weights for P6_0 and P7_0 to P6_1
    p6_w1 = self.p6_w1_relu(self.p6_w1) # torch.Size([2])
    weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon) # torch.Size([2])
    # Connections for P6_0 and P7_0 to P6_1 respectively, torch.Size([b, 112, 12, 12])
    p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))
    # Weights for P5_0 and P6_1 to P5_1
    p5_w1 = self.p5_w1_relu(self.p5_w1) # torch.Size([2])
    weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon) # torch.Size([2])
    # Connections for P5_0 and P6_1 to P5_1 respectively, torch.Size([b, 112, 24, 24])
    p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_up)))
    # Weights for P4_0 and P5_1 to P4_1
    p4_w1 = self.p4_w1_relu(self.p4_w1) # torch.Size([2])
    weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon) # torch.Size([2])
    # Connections for P4_0 and P5_1 to P4_1 respectively, torch.Size([b, 112, 48, 48])
    p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_up)))
    # Weights for P3_0 and P4_1 to P3_2
    p3_w1 = self.p3_w1_relu(self.p3_w1) # torch.Size([2])
    weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon) # torch.Size([2])
    # Connections for P3_0 and P4_1 to P3_2 respectively, torch.Size([b, 112, 96, 96])
    p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_up)))
    if self.first_time:
        p4_in = self.p4_down_channel_2(p4) # torch.Size([b, 112, 48, 48])
        p5_in = self.p5_down_channel_2(p5) # torch.Size([b, 112, 24, 24])
    # Weights for P4_0, P4_1 and P3_2 to P4_2
    p4_w2 = self.p4_w2_relu(self.p4_w2) # torch.Size([3])
    weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon) # torch.Size([3])
    # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively, torch.Size([b, 112, 48, 48])
    p4_out = self.conv4_down(
        self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))
    # Weights for P5_0, P5_1 and P4_2 to P5_2
    p5_w2 = self.p5_w2_relu(self.p5_w2) # torch.Size([3])
    weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon) # torch.Size([3])
    # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively, torch.Size([b, 112, 24, 24])
    p5_out = self.conv5_down(
        self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))
    # Weights for P6_0, P6_1 and P5_2 to P6_2
    p6_w2 = self.p6_w2_relu(self.p6_w2) # torch.Size([3])
    weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon) # torch.Size([3])
    # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
    p6_out = self.conv6_down(# torch.Size([b, 112, 12, 12])
        self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))
    # Weights for P7_0 and P6_2 to P7_2
    p7_w2 = self.p7_w2_relu(self.p7_w2) # torch.Size([3])
    weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon) # torch.Size([3])
    # Connections for P7_0 and P6_2 to P7_2, torch.Size([b, 112, 6, 6])
    p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))
    return p3_out, p4_out, p5_out, p6_out, p7_out

imgs = data['img'] # torch.Size([1, 3, 768, 768])
annot = data['annot'] # torch.Size([1, 3, 5])
#  EfficientNet, torch.Size([b, 48, 96, 96]), torch.Size([b, 120, 48, 48]), torch.Size([b, 352, 24, 24])
_, p3, p4, p5 = self.backbone_net(inputs)
features = (p3, p4, p5)
# torch.Size([b, 112, 96, 96]), torch.Size([b, 112, 48, 48]), 
# torch.Size([b, 112, 24, 24]), torch.Size([b, 112, 12, 12]), torch.Size([b, 112, 6, 6])
features = self.bifpn(features)
# 96*96+48*48+24*24+12*12+6*6=12276, 12276*9=110484, torch.Size([b, 110484, 4])
regression = self.regressor(features)
classification = self.classifier(features) # torch.Size([b, 110484, 90]), num_cls=90
anchors = self.anchors(inputs, inputs.dtype) # torch.Size([b, 110484, 4])
self.criterion = FocalLoss()