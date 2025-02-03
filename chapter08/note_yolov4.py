import cv2
image = images[0][0]*255
image = image.numpy().astype(np.uint8)
bbox = bboxes[0].numpy().astype(np.int_)
for b in bbox:
  cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), 255, 3)
cv2.imwrite('test.png', image)


def forward(self, x): # torch.Size([b, 3, 608, 608])
    ind = -2
    self.loss = None
    outputs = dict()
    out_boxes = []
    for block in self.blocks:
        ind = ind + 1
        # if ind > 0:
        #    return x

        if block['type'] == 'net':
            continue
        elif block['type'] in ['convolutional', 'maxpool', 'reorg', 'upsample', 'avgpool', 'softmax', 'connected']:
            x = self.models[ind](x)
            outputs[ind] = x
        elif block['type'] == 'route':
            layers = block['layers'].split(',')
            layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
            if len(layers) == 1:
                if 'groups' not in block.keys() or int(block['groups']) == 1:
                    x = outputs[layers[0]]
                    outputs[ind] = x
                else:
                    groups = int(block['groups'])
                    group_id = int(block['group_id'])
                    _, b, _, _ = outputs[layers[0]].shape
                    x = outputs[layers[0]][:, b // groups * group_id:b // groups * (group_id + 1)]
                    outputs[ind] = x
            elif len(layers) == 2:
                x1 = outputs[layers[0]]
                x2 = outputs[layers[1]]
                x = torch.cat((x1, x2), 1)
                outputs[ind] = x
            elif len(layers) == 4:
                x1 = outputs[layers[0]]
                x2 = outputs[layers[1]]
                x3 = outputs[layers[2]]
                x4 = outputs[layers[3]]
                x = torch.cat((x1, x2, x3, x4), 1)
                outputs[ind] = x
            else:
                print("rounte number > 2 ,is {}".format(len(layers)))

        elif block['type'] == 'shortcut':
            from_layer = int(block['from'])
            activation = block['activation']
            from_layer = from_layer if from_layer > 0 else from_layer + ind
            x1 = outputs[from_layer]
            x2 = outputs[ind - 1]
            x = x1 + x2
            if activation == 'leaky':
                x = F.leaky_relu(x, 0.1, inplace=True)
            elif activation == 'relu':
                x = F.relu(x, inplace=True)
            outputs[ind] = x
        elif block['type'] == 'sam':
            from_layer = int(block['from'])
            from_layer = from_layer if from_layer > 0 else from_layer + ind
            x1 = outputs[from_layer]
            x2 = outputs[ind - 1]
            x = x1 * x2
            outputs[ind] = x
        elif block['type'] == 'region':
            continue
            if self.loss:
                self.loss = self.loss + self.models[ind](x)
            else:
                self.loss = self.models[ind](x)
            outputs[ind] = None
        elif block['type'] == 'yolo':
            # if self.training:
            #     pass
            # else:
            #     boxes = self.models[ind](x)
            #     out_boxes.append(boxes)
            boxes = self.models[ind](x)
            out_boxes.append(boxes)
        elif block['type'] == 'cost':
            continue
        else:
            print('unknown type %s' % (block['type']))

    if self.training:
        return out_boxes
    else:
        return get_region_boxes(out_boxes)



# x: torch.Size([b, 3, 608, 608])
# CBM: outputs[0], torch.Size([b, 32, 608, 608])
## CSP1
# CBM: outputs[1, 3], torch.Size([b, 64, 304, 304]), CSP下采样
# CBM: outputs[2], torch.Size([b, 64, 304, 304])
# CBM: outputs[4], torch.Size([b, 64, 304, 304])
# CBM: outputs[5], torch.Size([b, 32, 304, 304])
# CBM: outputs[6], torch.Size([b, 64, 304, 304])
# CBM: outputs[7], torch.Size([b, 64, 304, 304])
# CBM: outputs[8], torch.Size([b, 64, 304, 304])
# concat: [outputs[8], outputs[2]], outputs[9], torch.Size([b, 128, 304, 304])
# CBM: outputs[10], torch.Size([b, 64, 304, 304])
# CSP2: outputs[23], torch.Size([b, 128, 152, 152])
# CSP8: outputs[54, 129], torch.Size([b, 256, 76, 76])
# CSP8: outputs[85, 119], torch.Size([b, 512, 38, 38])
# CSP4: outputs[104], torch.Size([b, 1024, 19, 19])
# CBL*3: outputs[107, 109, 111], torch.Size([b, 512, 19, 19])
## SPP
# MaxPool2d: kernel_size=5, stride=1, outputs[108], torch.Size([b, 512, 19, 19])
# MaxPool2d: kernel_size=9, stride=1, outputs[110], torch.Size([b, 512, 19, 19])
# MaxPool2d: kernel_size=13, stride=1, outputs[112], torch.Size([b, 512, 19, 19])
# concat: [outputs[112], outputs[110], outputs[113]], outputs[107], torch.Size([b, 2048, 19, 19])
# CBL*3: outputs[116], torch.Size([b, 512, 19, 19])
# CBL: outputs[117], torch.Size([b, 256, 19, 19])
# Upsample: outputs[118], torch.Size([b, 256, 38, 38])
# CBL: outputs[120], torch.Size([b, 256, 38, 38])
# concat: [outputs[120], outputs[118]], outputs[121], torch.Size([b, 512, 38, 38])
# CBL*5: outputs[126], torch.Size([b, 256, 38, 38])

# CBL: outputs[127], torch.Size([b, 128, 38, 38])
# Upsample: outputs[128], torch.Size([b, 128, 76, 76])
# CBL: outputs[130], torch.Size([b, 128, 76, 76])
# concat: [outputs[130], outputs[128]], outputs[131], torch.Size([b, 256, 76, 76])
# CBL*5: outputs[136, 140], torch.Size([b, 128, 76, 76])
# CBL: outputs[137], torch.Size([b, 256, 76, 76])
# Conv: outputs[138], out1, torch.Size([b, 255, 76, 76]), boxes-train

# CBL: outputs[136]->outputs[141], torch.Size([b, 256, 38, 38])
# concat: [outputs[141], outputs[126]], outputs[142], torch.Size([b, 512, 38, 38])
# CBL*5: outputs[147, 151], torch.Size([b, 256, 38, 38])
# CBL: outputs[148], torch.Size([b, 512, 38, 38])
# Conv: outputs[149], out2, torch.Size([b, 255, 38, 38]), boxes-train

# CBL: outputs[147]->outputs[152], torch.Size([b, 512, 19, 19])
# concat: [outputs[152], outputs[116]], outputs[153], torch.Size([b, 1024, 19, 19])
# CBL*5: outputs[158], torch.Size([b, 512, 19, 19])
# CBL: outputs[159], torch.Size([b, 1024, 19, 19])
# Conv: outputs[160], out3, torch.Size([b, 255, 19, 19]), boxes-train



if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
    area_c = torch.prod(con_br - con_tl, 2)  # convex area
    return iou - (area_c - area_u) / area_c  # GIoU
if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
    # convex diagonal squared
    c2 = torch.pow(con_br - con_tl, 2).sum(dim=2) + 1e-16
    if DIoU:
        return iou - rho2 / c2  # DIoU
    elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
        v = (4 / math.pi ** 2) * torch.pow(torch.atan(w1 / h1).unsqueeze(1) - torch.atan(w2 / h2), 2)
        with torch.no_grad():
            alpha = v / (1 - iou + v)
        return iou - (rho2 / c2 + v * alpha)  # CIoU

tgt_mask = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 4 + self.n_classes).to(device=self.device) # torch.Size([b, 3, 76, 76, 84])，与真实目标匹配的特征位置
obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).to(device=self.device) # torch.Size([b, 3, 76, 76])，与真实框重叠小于等于0.5，忽略ignore
tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).to(self.device) # torch.Size([b, 3, 76, 76, 2])，尺度
target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).to(self.device) # torch.Size([b, 3, 76, 76, 85]), [sigmoid(x), sigmoid(y), tw, th cls]


def forward(self, xin, labels=None):
    loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2 = 0, 0, 0, 0, 0, 0
    for output_id, output in enumerate(xin):
        batchsize = output.shape[0] # torch.Size([b, 255, 76, 76])
        fsize = output.shape[2] # 76
        n_ch = 5 + self.n_classes # 85, [x, y, w, h, obj, cls]

        output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize) # torch.Size([b, 3, 85, 76, 76])
        output = output.permute(0, 1, 3, 4, 2)  # torch.Size([b, 3, 76, 76, 85])

        # logistic activation for xy, obj, cls
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]]) # torch.Size([b, 3, 76, 76, 85])
        # 预测解码
        pred = output[..., :4].clone() # torch.Size([b, 3, 76, 76, 4]), [x, y, w, h]
        pred[..., 0] += self.grid_x[output_id] # 转为特征图上坐标，torch.Size([b, 3, 76, 76])
        pred[..., 1] += self.grid_y[output_id] # 转为特征图上坐标，torch.Size([b, 3, 76, 76])
        pred[..., 2] = torch.exp(pred[..., 2]) * self.anchor_w[output_id] # 预测宽度，torch.Size([b, 3, 76, 76])
        pred[..., 3] = torch.exp(pred[..., 3]) * self.anchor_h[output_id] # 预测高度，torch.Size([b, 3, 76, 76])

        # 真实框编码
        # obj_mask: torch.Size([b, 3, 76, 76])，正负样本，与真实框重叠小于等于0.5，或有真实目标匹配、
        # tgt_mask: torch.Size([b, 3, 76, 76, 84])，与真实目标匹配的特征位置
        # tgt_scale: torch.Size([b, 3, 76, 76, 2])，尺度
        # target: torch.Size([b, 3, 76, 76, 85]), [sigmoid(x), sigmoid(y), tw, th cls]
        obj_mask, tgt_mask, tgt_scale, target = self.build_target(pred, labels, batchsize, fsize, n_ch, output_id)
        
        # loss calculation
        output[..., 4] *= obj_mask # 计算负样本与真实样本处置信度损失
        output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask # 与真实目标相匹配的输出
        output[..., 2:4] *= tgt_scale # 尺度平衡因子，相当于权重

        target[..., 4] *= obj_mask
        target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
        target[..., 2:4] *= tgt_scale # 尺度平衡因子，相当于权重

        loss_xy += F.binary_cross_entropy(input=output[..., :2], target=target[..., :2],
                                            weight=tgt_scale * tgt_scale, reduction='sum') # 中心坐标损失
        loss_wh += F.mse_loss(input=output[..., 2:4], target=target[..., 2:4], reduction='sum') / 2 # 宽高损失
        loss_obj += F.binary_cross_entropy(input=output[..., 4], target=target[..., 4], reduction='sum') # 置信度损失
        loss_cls += F.binary_cross_entropy(input=output[..., 5:], target=target[..., 5:], reduction='sum') # 分类损失
        loss_l2 += F.mse_loss(input=output, target=target, reduction='sum') # 直接计算全部l2损失
    loss = loss_xy + loss_wh + loss_obj + loss_cls
    return loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2










