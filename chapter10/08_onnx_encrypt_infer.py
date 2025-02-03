import os
import cv2
import onnx
import numpy as np
import onnxruntime as ort
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# 指定 AES 密钥和 IV
key = b"1234567890abcdef"  # 16字节密钥，指定一个固定的密钥
iv = b"abcdef1234567890"  # 16字节初始化向量，指定一个固定的IV

# 解密 ONNX 模型文件
def decrypt_onnx_model(encrypted_data, cipher):
    # 使用解密器进行解密
    decryptor = cipher.decryptor()
    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
    
    # 去掉填充
    unpadder = padding.PKCS7(128).unpadder()
    unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()
    
    # 反序列化为 ONNX 模型
    model = onnx.ModelProto()
    model.ParseFromString(unpadded_data)

    return model

# 从加密文件读取并解密模型
def load_and_decrypt_model(encrypted_model_path):
    with open(encrypted_model_path, 'rb') as f:
        encrypted_data = f.read()

    # 使用指定的密钥和 IV
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())

    # 解密模型
    decrypted_model = decrypt_onnx_model(encrypted_data, cipher)
    return decrypted_model

# 进行推理
def infer_model(decrypted_model, input_data):
    # 使用 ONNX Runtime 进行推理
    # 使用 ONNX Runtime 加载解密后的模型
    session = ort.InferenceSession(decrypted_model.SerializeToString())
    
    # 获取模型输入名称
    input_name = session.get_inputs()[0].name

    # 进行推理
    result = session.run(None, {input_name: input_data})
    
    return result

# 主程序：加载加密模型，解密并进行推理
if __name__ == "__main__":
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    print("当前文件所在目录:", current_dir)
    # 创建结果保存目录
    save_dir = os.path.join(current_dir, 'result')
    os.makedirs(save_dir, exist_ok=True)

    encrypted_model_path = os.path.join(save_dir, "encrypted_model.bin")  # 你的加密模型路径

    # 加载并解密模型
    decrypted_model = load_and_decrypt_model(encrypted_model_path)
    
    # 加载ONNX模型
    session = ort.InferenceSession(decrypted_model.SerializeToString())

    # 读取图像
    image_path = "./data/helmet_data/test/images/hard_hat_workers116.png"
    image = cv2.imread(image_path)

    # 获取原图尺寸
    original_height, original_width = image.shape[:2]

    # 将图像大小调整为模型的输入大小（例如640x640）
    input_size = 640
    image_resized = cv2.resize(image, (input_size, input_size))

    # 归一化并转换为BGR -> RGB
    image_normalized = image_resized[:, :, ::-1] / 255.0

    # 转换为模型输入格式 (NCHW) 并增加批量维度
    input_tensor = np.expand_dims(image_normalized, axis=0).transpose(0, 3, 1, 2).astype(np.float32)

    # 打印输入形状，确保是 (1, 3, 640, 640)
    print(f"Input tensor shape: {input_tensor.shape}")

    # 获取模型输入和输出的名称
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    print("Model Inputs:", input_name)
    print("Model Outputs:", output_names)

    # 进行推理
    outputs = session.run(output_names, {input_name: input_tensor})

    # 打印输出的形状，确保正确, cx, cy, w, h, scores, classes
    print(f"Output shape: {outputs[0].shape}")
    print(f"Output dtype: {outputs[0].dtype}")

    # 假设outputs[0]是一个形状为 (1, 25200, N) 的张量
    output = outputs[0]  # shape (1, 25200, N)
    # print(output)

    # 提取边界框 (x1, y1, x2, y2)
    boxes = output[..., :4]

    # 提取置信度
    scores = output[..., 4:5]
    # print(scores)
    # print(np.sum(scores))

    # 提取类别ID
    class_ids = output[..., 5:]

    # 设置置信度阈值和NMS阈值
    confidence_threshold = 0.5
    nms_threshold = 0.4

    # 非极大值抑制 (NMS)
    # 需要将框、分数和类别ID转化为列表
    boxes_list = boxes.reshape(-1, 4).tolist()
    scores_list = scores.reshape(-1).tolist()
    class_ids_list = class_ids.reshape(-1).tolist()

    # 使用cv2的NMSBoxes进行非极大值抑制
    indices = cv2.dnn.NMSBoxes(boxes_list, scores_list, confidence_threshold, nms_threshold)
    # print(indices)
    # 在原图上绘制检测框
    for i in indices.flatten():
        # 获取边界框坐标和置信度
        x, y, w, h = boxes_list[i]
        score = scores_list[i]
        class_id = int(class_ids_list[i])

        # 计算缩放比例
        scale_x = original_width / input_size
        scale_y = original_height / input_size

        # 按比例调整框坐标
        x1 = (x - w / 2) * scale_x
        x2 = (x + w / 2) * scale_x
        y1 = (y - h / 2) * scale_y
        y2 = (y + h / 2) * scale_y

        # 绘制矩形框
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # 添加类别和置信度文本
        cv2.putText(image, f'{class_id} {score:.2f}', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 保存处理后的图像
    cv2.imwrite(f'{save_dir}/onnx_encrypt_infer.jpg', image)

    # 可选：显示处理后的图像
    # cv2.imshow('Detection', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
