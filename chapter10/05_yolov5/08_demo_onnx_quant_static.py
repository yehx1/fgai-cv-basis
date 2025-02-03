import cv2
import numpy as np
import onnx
from onnxruntime.quantization import QuantType, quantize_static, CalibrationDataReader
from pathlib import Path
from tqdm import tqdm


# ----------- 1. 定义 letterbox 函数 ------------
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    对图像进行 letterbox 处理，保持长宽比缩放并填充空白区域。
    只做演示使用，可根据实际需求灵活调整。
    
    :param im:        原图 (H, W, C)
    :param new_shape: 新尺寸 (height, width)，例如 (640, 640)
    :param color:     填充颜色
    :return:          letterbox 后的图像 (H_new, W_new, C)
    """
    # 原图的 (height, width)
    shape = im.shape[:2]
    # 计算最小缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # 缩放后的实际尺寸 (width, height)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    
    # 计算需要填充的尺寸
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding
    
    # 分成上下左右四边进行填充
    dw /= 2
    dh /= 2
    
    # 缩放
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # 四边需要填充的像素数
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # 使用指定颜色填充
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im


# ----------- 2. 主要参数配置 ------------
# 输入的 ONNX 模型路径
onnx_model_path = "runs/train/exp5/weights/best.onnx"
# 校准数据集路径（图像文件夹）
dataset_path = "../data/helmet_data/test/images"
# 输出的量化模型路径
quantized_model_path = "runs/train/exp5/weights/quant.onnx"


# ----------- 3. 自定义校准数据读取器 -----------
class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, dataset_path, input_size=640):
        self.dataset_path = dataset_path
        self.all_image_paths = [str(p) for p in Path(dataset_path).glob('*.*') 
                                if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
        self.all_image_paths = self.all_image_paths[:50]
        self.current_image_paths = list(self.all_image_paths)
        self.input_size = input_size

        # 读取 ONNX，获取模型的输入名
        model = onnx.load(onnx_model_path)
        self.input_name = model.graph.input[0].name

        # 进度条
        self.progress_bar = tqdm(total=len(self.all_image_paths),
                                 desc="Processing images",
                                 ncols=100)

    def get_next(self):
        # 每次取一张图做预处理
        while self.current_image_paths:
            image_path = self.current_image_paths.pop(0)
            image_tensor = self.load_image(image_path)
            if image_tensor is not None:
                self.progress_bar.update(1)
                return {self.input_name: image_tensor}
        return None

    def rewind(self):
        # 需要重复校准时重置
        self.current_image_paths = list(self.all_image_paths)
        self.progress_bar.reset()

    def load_image(self, image_path):
        """
        读取图像并进行 letterbox、BGR->RGB、归一化、维度转换等
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # 1) 使用 letterbox 保持长宽比缩放到 (self.input_size, self.input_size)
        #    填充色 114,114,114 是 YOLOv5 默认值
        image = letterbox(image, (self.input_size, self.input_size), color=(114, 114, 114))
        
        # 2) BGR -> RGB，并归一化到 [0,1]
        image = image[:, :, ::-1] / 255.0
        
        # 3) [H, W, C] -> [1, C, H, W] 并转换为 float32
        input_tensor = np.expand_dims(image, axis=0).transpose(0, 3, 1, 2).astype(np.float32)
        return input_tensor


# ----------- 4. 执行静态量化 -----------
if __name__ == "__main__":
    # 创建校准数据读取器
    calibration_reader = MyCalibrationDataReader(dataset_path)

    # 进行静态量化（后训练量化）
    quantize_static(
        onnx_model_path,           # 输入模型
        quantized_model_path,      # 量化后输出模型
        calibration_reader,        # 校准数据读取器
        weight_type=QuantType.QInt8,
        reduce_range=False,
        # per_channel=True,
        op_types_to_quantize= ['Conv'],
        # 如果你需要跳过某些算子，也可以设置 nodes_to_exclude
        nodes_to_exclude=['/model.0/conv/Conv']
    )

    print(f"量化模型已保存到 {quantized_model_path}")
