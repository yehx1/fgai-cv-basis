import os
import json
import zipfile

# 打包模型和配置
def create_package(model_path, config_path, output_zip):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        zipf.write(model_path, os.path.basename(model_path))
        zipf.write(config_path, os.path.basename(config_path))
    print(f"Model and config packed into {output_zip}")
# 解压模型包
def extract_package(package_path, extract_to):
    if not os.path.exists(package_path):
        raise FileNotFoundError(f"Package file not found: {package_path}")
    
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(package_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Package extracted to {extract_to}")
# 加载配置文件
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

# 主程序入口
if __name__ == '__main__':
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    print("当前文件所在目录:", current_dir)
    
    # 创建结果保存目录
    save_dir = os.path.join(current_dir, 'result')
    os.makedirs(save_dir, exist_ok=True)

    # # 设置训练目录和模型路径
    # model_save_dir = os.path.join(save_dir, 'extracted_model')  # 解压路径
    # os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'encrypted_model.bin')
    config_path = 'config.json'

    # 打包模型和配置
    zip_file = os.path.join(save_dir, 'model_package_with_config.zip')
    create_package(model_path, config_path, zip_file)

    # 演示解压并加载模型与配置
    extract_to = os.path.join(save_dir, 'unpacked_model')
    extract_package(zip_file, extract_to)

    # 加载解压后的配置
    loaded_config = load_config(os.path.join(extract_to, 'config.json'))

    print(f"Loaded config: {loaded_config}")
