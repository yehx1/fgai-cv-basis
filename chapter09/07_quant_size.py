import os
import torch
import torchvision

model_quantized = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=True)
model = torchvision.models.mobilenet_v2(pretrained=True)

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')

print_model_size(model)
print_model_size(model_quantized)

# 输出结果
# 14.24 MB
# 3.62 MB