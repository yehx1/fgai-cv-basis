# 《人工智能计算机视觉基础与实践指南》
## 第6章 目标检测网络
### 程序说明
#### 1. 01_yolo2coco.py
- YOLO格式的COCO128数据集转为COCO格式。
#### 2. 02_rcnn_coco128.py
- RCNN模型+COCO128数据集训练推理程序。注意该程序仅演示RCNN过程，实际训练过程更加复杂。
#### 3. 03_fast_rcnn_coco128.py
- Fast RCNN模型+COCO128数据集训练推理程序。
- 本项目训练模型保存于02_models/fgai_trained_models/chapter06/fast_rcnn_final.pth。
#### 4. 04_faster_rcnn_coco128.py
- Faster RCNN模型+COCO128数据集训练推理程序。
- 本项目训练模型保存于02_models/fgai_trained_models/chapter06/faster_rcnn_final.pth。
#### 5. 05_roi_pool_test.py:
- ROI Pool测试程序。
#### 6. 06_roi_align_test.py
- ROI Align测试程序。
#### 7. 07_roi_align_bilinear_interpolate.py
- ROI Align验证程序。
#### 8. 08_ssd_coco128.py
- SSD模型+COCO128数据集训练推理程序。
- note_ssd.py为部分关键程序解析。
- 本项目训练模型保存于02_models/fgai_trained_models/chapter06/ssd300_vgg16.pth。
#### 9. 09_pytorch-YOLO-v1
- YOLOv1模型+COCO128数据集训练推理程序。
- note_yolov1.py为部分关键程序解析。
- [详细使用说明：09_pytorch-YOLO-v1/README.md](09_pytorch-YOLO-v1/README.md)
#### 10. 10_yolo2-pytorch
- YOLOv2模型+COCO128数据集训练推理程序。
- note_yolov2.py为部分关键程序解析。
- [详细使用说明：10_yolo2-pytorch/README.md](10_yolo2-pytorch/README.md)
#### 11. 11_Yet-Another-EfficientDet-Pytorch
- EfficientDet-d2训练推理程序。
- note_efficientdet.py为部分关键程序解析。
- [详细使用说明：11_Yet-Another-EfficientDet-Pytorch/README.md](11_Yet-Another-EfficientDet-Pytorch/README.md)
#### 12. 12_efficientdet-pytorch
- Efficientdet-d0训练推理程序。
- note_efficientdet.py为部分关键程序解析。
- [详细使用说明：12_efficientdet-pytorch/README.md](12_efficientdet-pytorch/README.md)
#### 13. 13_pytorch-retinanet
- retinanet训练推理程序。
- note_retinanet.py为部分关键程序解析。
- [详细使用说明：13_pytorch-retinanet/README.md](13_pytorch-retinanet/README.md)
