import os
import onnx
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

# 定义数据读取器
class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, calibration_dataset):
        self.dataset = calibration_dataset
        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(self.dataset)
        return next(self.enum_data, None)

# 模拟一个数据集，用于校准（这里使用随机数据作为示例）
def create_calibration_dataset(batch_size=32, input_shape=(1, 3, 224, 224), num_batches=100):
    """模拟一个校准数据集"""
    dataset = []
    for _ in range(num_batches):
        # 创建一个随机的输入张量作为校准数据
        input_data = np.random.rand(batch_size, *input_shape).astype(np.float32)
        dataset.append({"input": input_data})
    return dataset

# 推理函数
def run_inference(model_path, input_data):
    """加载模型并运行推理"""
    # 创建 ONNX Runtime 会话
    session = ort.InferenceSession(model_path)

    # 获取模型的输入名称（假设只有一个输入）
    input_name = session.get_inputs()[0].name

    # 执行推理
    output = session.run(None, {input_name: input_data})

    return output

if __name__ == '__main__':
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    print("当前文件所在目录:", current_dir)
    # 创建结果保存目录
    save_dir = os.path.join(current_dir, 'result')
    os.makedirs(save_dir, exist_ok=True)

    # 假设已有 FP32 ONNX 模型 model_fp32.onnx
    model_fp32_path = f"{save_dir}/resnet18.onnx"
    model_int8_path = f"{save_dir}/resnet18_int8.onnx"

    # 如果 model_fp32_path 不存在，加载并保存一个示例模型
    if not os.path.exists(model_fp32_path):
        # 生成一个简单的示例 ONNX 模型
        # 假设此处你已经有了预训练的 resnet18 或其他模型
        import torch
        import torchvision.models as models

        # 创建一个 ResNet18 模型并转化为 ONNX 格式
        model = models.resnet18(pretrained=True)
        model.eval()

        # 随机生成一个输入
        dummy_input = torch.randn(1, 3, 224, 224)

        # 导出为 ONNX 格式
        torch.onnx.export(model, dummy_input, model_fp32_path, opset_version=11)

    print(f"FP32 模型路径: {model_fp32_path}")

    # 创建数据读取器实例
    calibration_data = create_calibration_dataset(batch_size=1, input_shape=(3, 224, 224), num_batches=100)  # 这里每批次只有 1 个样本
    data_reader = MyCalibrationDataReader(calibration_data)

    # 执行静态量化
    quantize_static(
        model_fp32_path,  # FP32 模型路径
        model_int8_path,  # 量化后的模型保存路径
        data_reader,      # 校准数据读取器
        quant_format=QuantType.QUInt8  # 可以选择 QUInt8 或 QInt8
    )

    print("量化后的模型已保存到:", model_int8_path)

    # 运行推理测试
    # 使用随机生成的数据作为输入
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)

    # 加载并运行量化后的 int8 模型
    output = run_inference(model_int8_path, input_data)

    print("推理输出:", output)
