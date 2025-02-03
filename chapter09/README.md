# 《人工智能计算机视觉基础与实践指南》
## 第9章 模型部署：模型转换、量化、加密、压缩打包等部署相关程序。
### 程序说明
#### 1. 01_export_onnx.py
- 导出ONNX模型
- 预训练模型官方下载地址：https://download.pytorch.org/models/resnet18-f37072fd.pth
- 本项目预训练模型存储路径：../02_models/pretrained_models/resnet18-f37072fd.pth
#### 2. 02_load_onnx.py
- ONNXmodel加载与推理
#### 3. 03_train_simplecnn.py
- 基于cifar-10训练的示例模型
#### 4. 04_export_simplecnn_onnx.py 
- 导出自定义ONNX模型
#### 5. 05_load_simplecnn_onnx.py
- 自定义ONNX模型加载与推理
#### 6. 06_optimize_onnx.py
- ONNX模型优化
#### 7. 07_quant_size.py
- 模型量化前后尺寸对比
#### 8. 08_torch_quant_static.py
- Pytorch静态量化
#### 9. 09_torch_quant_dynamic
- Pytorch动态量化
#### 10. 10_torch_quant_qat.py
- Pytorch QAT量化
#### 11. 11_onnx_ptq.py
- ONNX模型静态量化
#### 12. 12_encrypt_model.py
- 模型加密
#### 13. 13_zip_model.py
- 模型压缩打包

