import os
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import onnx

# 加载ONNX模型
def load_onnx_model(onnx_path):
    model = onnx.load(onnx_path)
    return model

# 加密ONNX模型
def encrypt_onnx_model(model, cipher):
    model_bytes = model.SerializeToString()  # 将模型序列化为字节流
    
    # 对字节流进行填充，使其符合块大小
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(model_bytes) + padder.finalize()

    # 使用加密器进行加密
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()

    # 打印加密前部分数据
    print("\nFirst 32 bytes of the model before encryption:")
    print(model_bytes[:32])  # 打印原始模型的前32字节

    # 打印加密后部分数据
    print("\nFirst 32 bytes of the encrypted model:")
    print(encrypted_data[:32])  # 打印加密后的模型前32字节

    return encrypted_data

# 解密ONNX模型
def decrypt_onnx_model(encrypted_data, cipher):
    # 使用解密器进行解密
    decryptor = cipher.decryptor()
    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
    
    # 去掉填充
    unpadder = padding.PKCS7(128).unpadder()
    unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()
    
    # 反序列化为ONNX模型
    model = onnx.ModelProto()
    model.ParseFromString(unpadded_data)

    # 打印解密后的前32字节
    print("\nFirst 32 bytes of the decrypted model (should match original):")
    print(unpadded_data[:32])  # 打印解密后的数据的前32字节

    return model

# 保存加密后的ONNX模型
def save_encrypted_onnx_model(encrypted_data, file_path):
    with open(file_path, 'wb') as f:
        f.write(encrypted_data)

# 保存解密后的ONNX模型
def save_decrypted_onnx_model(model, file_path):
    onnx.save(model, file_path)

# 主程序
if __name__ == "__main__":
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    print("当前文件所在目录:", current_dir)
    # 创建结果保存目录
    save_dir = os.path.join(current_dir, 'result')
    os.makedirs(save_dir, exist_ok=True)

    # 生成一个 16 字节的 AES 密钥 (应保密)
    # os.urandom(16)
    # print("Generated AES key:", key.hex())
    # os.urandom(16)
    # print("Generated IV:", iv.hex())

    # 指定 AES 密钥和 IV
    key = b"1234567890abcdef"  # 16字节密钥，指定一个固定的密钥
    iv = b"abcdef1234567890"  # 16字节初始化向量，指定一个固定的IV
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())

    # 加载ONNX模型
    onnx_model_path = "yolov5/runs/train/exp5/weights/quant.onnx"  # 替换为你的ONNX模型路径
    model = load_onnx_model(onnx_model_path)
    print("Loaded ONNX model.")

    # 加密ONNX模型
    encrypted_data = encrypt_onnx_model(model, cipher)
    encrypted_model_path = os.path.join(save_dir, "encrypted_model.bin")
    save_encrypted_onnx_model(encrypted_data, encrypted_model_path)
    print(f"Encrypted ONNX model saved to {encrypted_model_path}")

    # 解密ONNX模型
    decrypted_model = decrypt_onnx_model(encrypted_data, cipher)
    decrypted_model_path = os.path.join(save_dir, "decrypted_model.onnx")
    save_decrypted_onnx_model(decrypted_model, decrypted_model_path)
    print(f"Decrypted ONNX model saved to {decrypted_model_path}")
