# 《人工智能计算机视觉基础与实践指南》
## 第10章 实验课程：安全帽检测模型转换、量化、加密、压缩打包、人脸识别业务等部署相关程序。
### 程序说明
#### 1. 01_kaggle_helmet_download.py
- kaggle安全帽数据集下载
- 网页下载地址：https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection/data
#### 2. 02_dataset_analysis.py
- 统计数据集中标签类别、数量等信息
#### 3. 03_voc2yolo.py
- VOC标签转换为YOLO格式
#### 4. 04_dataset_split
- 划分数据集：训练集、验证集、测试集
#### 5. 05_yolov5
- chapter10/05_yolov5/01_train.py：训练模型
- chapter10/05_yolov5/02_detect.py：批量推理
- chapter10/05_yolov5/03_demo.py：单张图片推理
- chapter10/05_yolov5/04_export.py：导出ONNX模型
- chapter10/05_yolov5/05_demo_onnx_infer.py： ONNX模型推理
- chapter10/05_yolov5/06_demo_onnx_quant_dynamic.py：ONNX模型动态量化
- chapter10/05_yolov5/07_demo_onnx_quant_dynamic_infer.py：ONNX模型动态量化推理
- chapter10/05_yolov5/08_demo_onnx_quant_static.py：ONNX模型静态量化
- chapter10/05_yolov5/09_demo_onnx_quant_static_infer.py：ONNX模型静态量化推理
#### 6. 06_encrypt_model.py
- 模型加密
#### 7. 07_onnx_decrypt_infer.py
- ONNX模型解密文件加载与推理
#### 8. 08_onnx_encrypt_infer.py
- ONNX模型加密文件加载与推理
#### 9. 09_zip_model.py
- 模型压缩打包
#### 10. 10_unzip_decrypt_infer.py
- 模型解压、解密、推理
#### 11. 11_cv2_rtsp.py
- RTSP摄像头读取与显示
#### 12. 12_face_checkin.py
- 人脸识别签到系统
