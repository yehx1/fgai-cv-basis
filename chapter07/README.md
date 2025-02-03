# 《人工智能计算机视觉基础与实践指南》
## 第7章 语义分割网络
### 程序说明
#### 1. 01_color_seg.py
- 基于颜色的分割。
- 示例效果路径：chapter07/result/color_segmented_image.jpg
#### 2. 02_gabor_seg.py
- 基于gabor的分割。
- 示例效果路径：chapter07/result/gabor_segmented_image.jpg
#### 3. 03_hog_seg.py
- 基于hog的分割。
- 示例效果路径：chapter07/result/hog_segmented_image.jpg
#### 4. 04_crf_seg.py
- 基于CRF的分割。
- 示例效果路径：chapter07/result/crf_segmented_image.jpg
#### 5. 05_fcn_deconv.py
- 反卷积测试。
#### 6. 06_fcn_voc.py
- 基于FCN的分割，包括训练与推理程序。
- note_fcn.py为部分关键程序解析。
- 训练模型保存路径：chapter07/result/fcn_voc2007_epoch_final.pth
- 推理图片示例路径：chapter07/result/fcn_infer_sample.jpg
- 本项目训练模型保存于../02_models/fgai_trained_models/chapter07/fcn_voc2007_epoch_final.pth。
#### 7. 07_deeplab_dilation_conv.py
- 空洞卷积测试。
#### 8. 08_DeepLab-V1-PyTorch
- 基于Deeplab V1的分割。
- note_deeplab_v1.py为部分关键程序解析。
- [详细使用说明：08_DeepLab-V1-PyTorch/README.md](08_DeepLab-V1-PyTorch/README.md)
#### 9. 09_deeplab-pytorch
- 基于Deeplab V2的分割。
- note_deeplab_v2.py为部分关键程序解析。
- [详细使用说明：09_deeplab-pytorch/README.md](09_deeplab-pytorch/README.md)
#### 10. 10_deeplab_v3_voc.py
- 基于Deeplab V3的分割。
- note_deeplab_v3.py为部分关键程序解析。
- 训练模型保存路径：chapter07/result/deeplabv3_voc2007_epoch_final.pth
- 推理图片示例路径：chapter07/result/deeplabv3_infer_sample.jpg
- 本项目训练模型保存于../02_models/fgai_trained_models/chapter07/deeplabv3_voc2007_epoch_final.pth。
#### 11. 11_DeepLabv3Plus-Pytorch
- 基于Deeplab V3+的分割。
- note_deeplab_v3_plus.py为部分关键程序解析。
- [详细使用说明：11_DeepLabV3Plus-Pytorch/README.md](11_DeepLabV3Plus-Pytorch/README.md)
#### 12. 12_Pytorch-UNet
- 基于UNet的分割。
- note_unet.py为部分关键程序解析。
- [详细使用说明：12_Pytorch-UNet/README.md](12_Pytorch-UNet/README.md)
#### 13. 13_segnet_voc.py
- 基于segnet的分割。
- note_segnet.py为部分关键程序解析。
- 训练模型保存路径：chapter07/result/segnet_voc2007_epoch_final.pth
- 推理图片示例路径：chapter07/result/segnet_infer_sample_0.jpg
- 本项目训练模型保存于../02_models/fgai_trained_models/chapter07/segnet_voc2007_epoch_final.pth。









