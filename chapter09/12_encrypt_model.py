import os
import torch
import torch.nn as nn
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)  # 一个简单的线性层

    def forward(self, x):
        return self.fc(x)


# 加密模型权重
def encrypt_model_weights(model, cipher):
    encrypted_weights = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            weight_data = param.data.numpy().tobytes()  # 转为字节
            padded_data = pad(weight_data, AES.block_size)  # 填充
            encrypted_data = cipher.encrypt(padded_data)  # 加密
            encrypted_weights[name] = encrypted_data
    return encrypted_weights


# 解密模型权重
def decrypt_model_weights(encrypted_weights, cipher):
    decrypted_weights = {}
    for name, enc_data in encrypted_weights.items():
        # 解密数据
        padded_data = cipher.decrypt(enc_data)
        # 去填充
        weight_data = unpad(padded_data, AES.block_size)

        # 根据解密后的数据长度恢复权重形状
        print(f"Decrypted {name}, length of data: {len(weight_data)// 4}")
        if 'weight' in name:
            shape = (5, 10)  # 10是fc层的输入维度
        elif 'bias' in name:
            shape = (5,)  # 偏置是大小为5的一维张量
        else:
            raise ValueError(f"Unexpected parameter: {name}")

        print(f"Expected shape for {name}: {shape}")
        weight_tensor = torch.tensor(np.frombuffer(weight_data, dtype=np.float32).reshape(shape))
        
        decrypted_weights[name] = weight_tensor
        print(f"Decrypted weight for {name}: {weight_tensor}")

    return decrypted_weights



if __name__ == "__main__":
    # 生成一个 16 字节的 AES 密钥 (应保密)
    key = os.urandom(16)
    cipher = AES.new(key, AES.MODE_CBC)

    # 创建模型实例
    model = SimpleModel()
    print("Model structure:", model)

    # 随机初始化模型权重
    print("\nOriginal weights (before encryption):")
    print("Model fc.weight:", model.fc.weight.data)
    print("Model fc.bias:", model.fc.bias.data)

    # 加密模型权重
    encrypted_weights = encrypt_model_weights(model, cipher)
    print("\nEncrypted Weights:")
    for name, enc_data in encrypted_weights.items():
        print(f"{name}: {enc_data[:32]}...")  # 打印前32个字节进行预览

    # 解密权重
    iv = os.urandom(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_weights = decrypt_model_weights(encrypted_weights, cipher)

    # 将解密的权重赋值给模型
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = decrypted_weights[name]

    print("\nDecrypted Model Weights:")
    print("Model fc.weight (after decryption):", model.fc.weight.data)
    print("Model fc.bias (after decryption):", model.fc.bias.data)
