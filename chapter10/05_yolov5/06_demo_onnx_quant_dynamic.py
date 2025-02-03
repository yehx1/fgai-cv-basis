from onnxruntime.quantization import QuantType, quantize_dynamic

# 输入的 ONNX 模型路径
onnx_model_path = "runs/train/exp5/weights/best.onnx"
# 输出的量化模型路径
quantized_model_path = "runs/train/exp5/weights/quantd.onnx"

# 使用 ONNX Runtime 进行后训练量化
quantize_dynamic(
    onnx_model_path,  # 输入模型路径
    quantized_model_path,  # 输出模型路径
    weight_type=QuantType.QUInt8  # 仅量化权重为 INT8 类型
)

print(f"量化模型已保存到 {quantized_model_path}")
