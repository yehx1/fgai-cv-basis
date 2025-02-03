def encoder(self,boxes,labels):
    '''
    boxes (tensor) [[x1,y1,x2,y2],[]]
    labels (tensor) [...]
    return 7x7x30
    '''
    grid_num = 7
    target = torch.zeros((grid_num,grid_num,30)) # torch.Size([14, 14, 30])
    cell_size = 1./grid_num
    wh = boxes[:,2:]-boxes[:,:2]
    cxcy = (boxes[:,2:]+boxes[:,:2])/2
    for i in range(cxcy.size()[0]):
        cxcy_sample = cxcy[i]
        ij = (cxcy_sample/cell_size).ceil()-1 # 中心点的网格坐标
        target[int(ij[1]),int(ij[0]),4] = 1 # 置信度为1
        target[int(ij[1]),int(ij[0]),9] = 1 # 置信度为1
        target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1 # label，独热码
        xy = ij*cell_size #匹配到的网格的左上角相对坐标
        # 对于真实框，两个候选框的预测目标一致
        # 中心点相对于网格的偏移, 特征图上绝对坐标偏移
        delta_xy = (cxcy_sample -xy)/cell_size
        target[int(ij[1]),int(ij[0]),2:4] = wh[i] # 第一个候选框宽度和长度比例
        target[int(ij[1]),int(ij[0]),:2] = delta_xy # 第一个候选框中心偏移
        target[int(ij[1]),int(ij[0]),7:9] = wh[i] # 第二个候选框宽度和长度比例
        target[int(ij[1]),int(ij[0]),5:7] = delta_xy # 第二个候选框中心偏移
    return target


def forward(self,pred_tensor,target_tensor):
'''
pred_tensor: (tensor) size(batchsize,S,S,Bx5+20=30) [x,y,w,h,c]
target_tensor: (tensor) size(batchsize,S,S,30)
'''
N = pred_tensor.size()[0]
coo_mask = target_tensor[:,:,:,4] > 0
noo_mask = target_tensor[:,:,:,4] == 0
coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor) # torch.Size([b, 14, 14, 30])
noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor) # torch.Size([b, 14, 14, 30])
coo_pred = pred_tensor[coo_mask].view(-1,30) # 有目标处的预测结果，torch.Size([N, 30])
box_pred = coo_pred[:,:10].contiguous().view(-1,5) # box[x1,y1,w1,h1,c1], torch.Size([2N, 5])
class_pred = coo_pred[:,10:] # 类别预测, torch.Size(N,20])
coo_target = target_tensor[coo_mask].view(-1,30) # torch.Size([N, 30])
box_target = coo_target[:,:10].contiguous().view(-1,5) # torch.Size([2N, 5])
class_target = coo_target[:,10:] # torch.Size([N, 20])
# compute not contain obj loss
noo_pred = pred_tensor[noo_mask].view(-1,30) # 没有目标的点，torch.Size([K, 30])
noo_target = target_tensor[noo_mask].view(-1,30) # torch.Size([K, 30])
noo_pred_mask = torch.cuda.ByteTensor(noo_pred.size()) # torch.Size([K, 30])
noo_pred_mask.zero_()
noo_pred_mask[:,4]=1;noo_pred_mask[:,9]=1
noo_pred_c = noo_pred[noo_pred_mask] # noo pred只需要计算c的损失 size[-1,2]，只计算有无目标，torch.Size([2k])
noo_target_c = noo_target[noo_pred_mask] # torch.Size([2k]，真实目标标签均为背景标签0
nooobj_loss = F.mse_loss(noo_pred_c,noo_target_c,size_average=False) # 背景损失，无目标置信度损失
#compute contain obj loss
coo_response_mask = torch.cuda.ByteTensor(box_target.size()) # torch.Size([2N, 5])
coo_response_mask.zero_()
coo_not_response_mask = torch.cuda.ByteTensor(box_target.size())
coo_not_response_mask.zero_() # torch.Size([2N, 5]
box_target_iou = torch.zeros(box_target.size()).cuda() # torch.Size([2K, 5]
for i in range(0,box_target.size()[0],2): #choose the best iou box
    box1 = box_pred[i:i+2]
    box1_xyxy = Variable(torch.FloatTensor(box1.size()))
    box1_xyxy[:,:2] = box1[:,:2]/14. - 0.5*box1[:,2:4]
    box1_xyxy[:,2:4] = box1[:,:2]/14. + 0.5*box1[:,2:4]
    box2 = box_target[i].view(-1,5)
    box2_xyxy = Variable(torch.FloatTensor(box2.size()))
    box2_xyxy[:,:2] = box2[:,:2]/14. - 0.5*box2[:,2:4]
    box2_xyxy[:,2:4] = box2[:,:2]/14. + 0.5*box2[:,2:4]
    iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]
    max_iou,max_index = iou.max(0) # torch.Size([2, 1])
    max_index = max_index.data.cuda() # 选择IOU较大检测框
    coo_response_mask[i+max_index]=1 # 两个预测框中跟相关的
    coo_not_response_mask[i+1-max_index]=1 # 两个预测框中不想管的
    box_target_iou[i+max_index,torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
box_target_iou = Variable(box_target_iou).cuda() # torch.Size([2N, 5])
#1.response loss
box_pred_response = box_pred[coo_response_mask].view(-1,5 ) # torch.Size([N, 5])
box_target_response_iou = box_target_iou[coo_response_mask].view(-1,5) # torch.Size([N, 5])
box_target_response = box_target[coo_response_mask].view(-1,5) # torch.Size([N, 5])
contain_loss = F.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False) # 包含目标损失
loc_loss = F.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],size_average=False) + F.mse_loss(torch.sqrt(box_pred_response[:,2:4]),torch.sqrt(box_target_response[:,2:4]),size_average=False) # 位置损失
#2.not response loss
box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5) # torch.Size([N, 5])
box_target_not_response = box_target[coo_not_response_mask].view(-1,5) # torch.Size([N, 5])
box_target_not_response[:,4]= 0 # 将剩余B-1个位置的目标设置为背景，即仅有一个有效的预测框
# 确保B个结果只有一个包含目标，实际上是不包含目标损失的一部分
not_contain_loss = F.mse_loss(box_pred_not_response[:,4], box_target_not_response[:,4],size_average=False)
#3.class loss
class_loss = F.mse_loss(class_pred,class_target,size_average=False) # 分类损失
return (self.l_coord*loc_loss + 2*contain_loss + not_contain_loss + self.l_noobj*nooobj_loss + class_loss)/N
