import os
import torch
import torchvision


if __name__ == '__main__':
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    print("当前文件所在目录:", current_dir)
    # 创建结果保存目录
    save_dir = os.path.join(current_dir, 'result')
    os.makedirs(save_dir, exist_ok=True)

    # Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, f"{save_dir}/resnet18.onnx", 
                    input_names=["input"], output_names=["output"])