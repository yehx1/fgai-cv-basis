import cv2
import numpy as np
import onnx
from onnxruntime.quantization import QuantType, quantize_static, CalibrationDataReader
from pathlib import Path
from tqdm import tqdm  # 导入 tqdm 库

# 输入的 ONNX 模型路径
onnx_model_path = "runs/train/exp5/weights/best.onnx"
# 校准数据集路径（图像文件夹）
dataset_path = "../data/helmet_data/test/images"
# 输出的量化模型路径
quantized_model_path = "runs/train/exp5/weights/quant.onnx"

# 自定义数据读取器，用于从图片文件夹读取数据进行校准
class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataset_path, input_size=640):
        self.dataset_path = dataset_path
        self.image_paths = [str(p) for p in Path(dataset_path).glob('*.png')]  # 假设是png图像，按需要调整
        self.input_size = input_size

        # 加载模型获取输入名称
        model = onnx.load(onnx_model_path)
        self.input_name = model.graph.input[0].name  # 自动获取第一个输入的名称

        # 使用 tqdm 显示进度
        self.progress_bar = tqdm(self.image_paths, desc="Processing images", ncols=100)

    def get_next(self):
        # 获取一个图像文件进行校准
        if self.image_paths:
            image_path = self.image_paths.pop(0)  # 从队列中取出一个图像路径
            image = self.load_image(image_path)  # 加载图像并转为模型输入格式
            self.progress_bar.update(1)  # 更新进度条
            return {self.input_name: image}  # 使用自动获取的输入名称
        return None

    def rewind(self):
        pass

    def load_image(self, image_path):
        # 使用 OpenCV 加载图像
        image = cv2.imread(image_path)
        
        # 将图像大小调整为模型的输入大小 (例如 640x640)
        image_resized = cv2.resize(image, (self.input_size, self.input_size))
        
        # 归一化并转换为 BGR -> RGB
        image_normalized = image_resized[:, :, ::-1] / 255.0
        
        # 转换为模型输入格式 (NCHW) 并增加批量维度
        input_tensor = np.expand_dims(image_normalized, axis=0).transpose(0, 3, 1, 2).astype(np.float32)
        return input_tensor

# 创建校准数据读取器
calibration_reader = MyCalibrationDataReader(dataset_path)

# https://github.com/microsoft/onnxruntime/issues/14233
# 进行静态量化（后训练量化）
quantize_static(
    onnx_model_path,  # 输入模型路径
    quantized_model_path,  # 输出量化后的模型路径
    calibration_reader,  # 校准数据读取器
    weight_type=QuantType.QInt8,
    # nodes_to_exclude=['Mul_214', 'Mul_225', 'Mul_249', 'Mul_260', 'Mul_284', 'Mul_295', 'Concat_231', 'Concat_266', 'Concat_301', 'Concat_303'],
    reduce_range=True
)

print(f"量化模型已保存到 {quantized_model_path}")
