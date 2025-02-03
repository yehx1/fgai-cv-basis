# Ultralytics YOLOv5 🚀, AGPL-3.0 license

# Parameters
nc: 80 # number of classes
depth_multiple: 0.33 # model depth multiple
width_multiple: 0.50 # layer channel multiple
anchors:
  - [10, 13, 16, 30, 33, 23] # P3/8
  - [30, 61, 62, 45, 59, 119] # P4/16
  - [116, 90, 156, 198, 373, 326] # P5/32

# YOLOv5 v6.0 backbone
backbone:
  # [from, number, module, args]
  [
    [-1, 1, Conv, [64, 6, 2, 2]], # 0-P1/2
    [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
    [-1, 3, C3, [128]],
    [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
    [-1, 6, C3, [256]],
    [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
    [-1, 9, C3, [512]],
    [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
    [-1, 3, C3, [1024]],
    [-1, 1, SPPF, [1024, 5]], # 9
  ]

           from  n    params  module                                  arguments                     
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]              
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]                
  2                -1  1     18816  models.common.C3                        [64, 64, 1]                   
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]               
  4                -1  2    115712  models.common.C3                        [128, 128, 2]                 
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]              
  6                -1  3    625152  models.common.C3                        [256, 256, 3]                 
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]              
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]                 
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]                 
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]              
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 12           [-1, 6]  1         0  models.common.Concat                    [1]                           
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]          
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]              
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']          
 16           [-1, 4]  1         0  models.common.Concat                    [1]                           
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]          
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]              
 19          [-1, 14]  1         0  models.common.Concat                    [1]                           
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]          
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]              
 22          [-1, 10]  1         0  models.common.Concat                    [1]                           
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]          
 24      [17, 20, 23]  1    229245  models.yolo.Detect                      [80, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
# YOLOv5 v6.0 head
head: [
    [-1, 1, Conv, [512, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [[-1, 6], 1, Concat, [1]], # cat backbone P4
    [-1, 3, C3, [512, False]], # 13

    [-1, 1, Conv, [256, 1, 1]],
    [-1, 1, nn.Upsample, [None, 2, "nearest"]],
    [[-1, 4], 1, Concat, [1]], # cat backbone P3
    [-1, 3, C3, [256, False]], # 17 (P3/8-small)

    [-1, 1, Conv, [256, 3, 2]],
    [[-1, 14], 1, Concat, [1]], # cat head P4
    [-1, 3, C3, [512, False]], # 20 (P4/16-medium)

    [-1, 1, Conv, [512, 3, 2]],
    [[-1, 10], 1, Concat, [1]], # cat head P5
    [-1, 3, C3, [1024, False]], # 23 (P5/32-large)

    [[17, 20, 23], 1, Detect, [nc, anchors]], # Detect(P3, P4, P5)
  ]



[4, 6, 10, 14, 17, 20, 23]
torch.Size([b, 3, 640, 640])
0: CBS, Conv(3, 32, 6, 2), torch.Size([b, 32, 320, 320])
1: CBS, Conv(32, 64, 3, 2), torch.Size([b, 64, 160, 160])
2: C3, torch.Size([b, 64, 160, 160])
3: CBS, Conv(64, 128, 3, 2), torch.Size([b, 128, 80, 80])
4: C3, torch.Size([b, 128, 80, 80])
5: CBS, Conv(128, 256, 3, 2), torch.Size([b, 256, 40, 40])
6: C3, torch.Size([b, 256, 40, 40])
7: CBS, Conv(256, 512, 3, 2), torch.Size([b, 512, 20, 20])
8: C3, torch.Size([b, 512, 20, 20])
9: SPPF, torch.Size([b, 512, 20, 20])
10: CBS, Conv(512, 256, 1, 1), torch.Size([b, 256, 20, 20])
11: Upsample, *2, torch.Size([b, 256, 40, 40])
12: concat[6, 11], torch.Size([b, 512, 40, 40])
13: C3, torch.Size([b, 256, 40, 40])
14: CBS, Conv(256, 128, 1, 1), torch.Size([b, 128, 40, 40])
15: Upsample, *2, torch.Size([b, 128, 80, 80])
16: concat[4, 15], torch.Size([b, 256, 80, 80])
17: C3, torch.Size([b, 128, 80, 80])
18: CBS, Conv(128, 128, 3, 2), torch.Size([b, 128, 40, 40])
19: concat[14, 18], torch.Size([b, 256, 40, 40])
20: C3, torch.Size([b, 256, 40, 40])
21: CBS, Conv(256, 256, 3, 2), torch.Size([b, 256, 20, 20])
22: concat[10, 21], torch.Size([b, 512, 20, 20])
23: C3, torch.Size([b, 512, 20, 20])
24: Detect: Conv(128, 255, 1, 1)、Conv(256, 255, 1, 1)、Conv(512, 255, 1, 1), torch.Size([b, 255, 80, 80]), torch.Size([b, 255, 40, 40]), torch.Size([b, 255, 20, 20])
output: [torch.Size([b, 3, 80, 80, 85]), torch.Size([b, 3, 40, 40, 85]), torch.Size([b, 3, 20, 20, 85])]

def build_targets(self, p, targets):
    """Prepares model targets from input targets (image,class,x,y,w,h) for loss computation, returning class, box,
    indices, and anchors.
    """
    na, nt = self.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

    g = 0.5  # bias
    off = ( # 向相邻网格偏移
        torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device,
        ).float()
        * g
    )  # offsets
    for i in range(self.nl):
        anchors, shape = self.anchors[i], p[i].shape
        gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain, tensor([ 1.,  1., 80., 80., 80., 80.,  1.])
        # Match targets to anchors
        t = targets * gain  # shape(3,n,7), 将归一化bbox坐标转换到特征图尺度
        if nt:
            # Matches
            r = t[..., 4:6] / anchors[:, None]  # wh ratio, 计算目标框与锚框的宽高比, torch.Size([3, n, 2])
            j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]  # compare, 删除宽高比小于4.0的anchor, torch.Size([3, n])
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter, torch.Size([nk, 7]), 真实目标匹配的anchor

            # Offsets
            gxy = t[:, 2:4]  # grid xy, torch.Size([nk, 2])
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1 < g) & (gxy > 1)).T # j 和 k: 判断目标框是否靠近网格左上方, torch.Size([nk])
            l, m = ((gxi % 1 < g) & (gxi > 1)).T # l 和 m: 判断目标框是否靠近网格右下方
            j = torch.stack((torch.ones_like(j), j, k, l, m)) # torch.Size([5, nk])
            t = t.repeat((5, 1, 1))[j] # torch.Size([k*5, nk])
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0
        # Define
        bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
        a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid indices
        # Append
        indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class
    return tcls, tbox, indices, anch

 def __call__(self, p, targets):  # predictions, targets
    """Performs forward pass, calculating class, box, and object loss for given predictions and targets."""
    lcls = torch.zeros(1, device=self.device)  # class loss
    lbox = torch.zeros(1, device=self.device)  # box loss
    lobj = torch.zeros(1, device=self.device)  # object loss
    tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
    # Losses
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj, torch.Size([b, 3, 80, 80])
        n = b.shape[0]  # number of targets
        if n:
            # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
            pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions
            # Regression
            pxy = pxy.sigmoid() * 2 - 0.5
            pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1)  # predicted box, torch.Size([n, 4])
            iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
            lbox += (1.0 - iou).mean()  # iou loss
            # Objectness
            iou = iou.detach().clamp(0).type(tobj.dtype)
            if self.sort_obj_iou:
                j = iou.argsort()
                b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
            if self.gr < 1:
                iou = (1.0 - self.gr) + self.gr * iou
            tobj[b, a, gj, gi] = iou  # iou ratio
            # Classification
            if self.nc > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(pcls, self.cn, device=self.device)  # targets, torch.Size([n, 80])
                t[range(n), tcls[i]] = self.cp
                lcls += self.BCEcls(pcls, t)  # 分类损失，BCELoss
            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
        obji = self.BCEobj(pi[..., 4], tobj) # 置信度损失与IOU相关
        lobj += obji * self.balance[i]  # obj loss，[4.0, 1.0, 0.4]，为不同尺度设置平衡权重
        if self.autobalance:
            self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
    if self.autobalance:
        self.balance = [x / self.balance[self.ssi] for x in self.balance]
    lbox *= self.hyp["box"] # 0.05
    lobj *= self.hyp["obj"] # 1.0
    lcls *= self.hyp["cls"] # 0.5
    bs = tobj.shape[0]  # batch size
    return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

https://blog.csdn.net/dou3516/article/details/130358680
https://www.cnblogs.com/Fish0403/p/17451698.html
https://github.com/ultralytics/yolov5/issues/6998#41

Lloc lbox CIOU
Lcls BCELOSS
默认没有使用Label Smoothing
Lobj BCE

lbox *= self.hyp["box"] # 0.05
lobj *= self.hyp["obj"] # 1.0
lcls *= self.hyp["cls"] # 0.5

