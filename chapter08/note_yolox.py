input: x, torch.Size([b, 3, 640, 640])
backbone: CSPDarknet + FPN, 
fpn_outs = self.backbone(x)

#主要目的是通过计算目标框的中心是否位于锚点的有效范围内，来筛选出符合条件的锚点。这样可以减少计算量，并避免不合适的锚点匹配，从而提高训练效率，节省GPU内存。

# 相当于在特征图上相邻网格内（3x3)，
# 思考：自适应半径、必须在目标上
center_radius = 1.5 # center_radius 设定了锚点的中心范围，这个值是 1.5，意味着锚点的有效区域是目标框中心周围的 1.5 * stride 范围。

#计算目标框的左右上下边界，并根据 center_dist 将目标框扩展。这里的 gt_bboxes_per_image[:, 0:1] 取的是目标框的左边界（xmin），gt_bboxes_per_image[:, 1:2] 取的是目标框的上边界（ymin）。然后通过 - center_dist 和 + center_dist 分别扩展目标框的左右和上下边界。

anchor_filter = is_in_centers.sum(dim=0) > 0 # 锚点至少对应一个真实目标，torch.Size([8400]), 假设有n个有效目标
geometry_relation = is_in_centers[:, anchor_filter] # 每个目标对应的锚点mask，torch.Size([k, n])

# 匹配候选框和GT框：根据代价和IoU值，通过动态选择匹配的候选框，使用 matching_matrix 记录匹配结果。
# 处理多个匹配：如果一个候选框与多个GT框匹配，它会选择代价最小的匹配。
# 更新前景框掩码：标记哪些候选框是前景框，并计算这些前景框的IoU和类别信息。