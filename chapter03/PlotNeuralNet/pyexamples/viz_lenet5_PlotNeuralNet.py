# 将该文件复制到PlotNeuralNet/pyexamples文件夹下
# 进入到lotNeuralNet/pyexamples文件夹后，运行 bash ../tikzmake.sh viz_lenet5_PlotNeuralNet
# 运行完成后，会在文件夹下生成pdf格式图片，viz_lenet5_PlotNeuralNet.pdf
import sys
sys.path.append('../')
from pycore.tikzeng import *

# 定义 LeNet-5 神经网络架构（不包含输入图片，可通过to_Input加载）
def lenet5_architecture():
    arch = [
        to_head('..'), 
        to_cor(),
        to_begin(),
        # 卷积层1: 6个卷积核，输出28x28x6
        to_Conv("C1", 28, 6, offset="(0,0,0)", to="(0,0,0)", height=28, depth=28, width=6, caption="Conv1"),
        # 池化层1: 最大池化，输出14x14x6
        to_Pool(name="S2", offset="(0,0,0)", to="(C1-east)", width=6, height=14, depth=14, opacity=0.5, caption="Pool1"),
        # 卷积层2: 16个卷积核，输出10x10x16
        to_Conv("C3", 10, 16, offset="(0,0,0)", to="(S2-east)", height=10, depth=10, width=16, caption="Conv2"),
        # 池化层2: 最大池化，输出5x5x16
        to_Pool(name="S4", offset="(0,0,0)", to="(C3-east)", width=16, height=5, depth=5, opacity=0.5, caption="Pool2"),
        # 全连接层1: 120个神经元
        to_Conv("C5", 120, 1, offset="(2,0,0)", to="(S4-east)", height=1, depth=120, width=1, caption="FC1"),
        to_connection("S4", "C5"),
        # 全连接层2: 84个神经元
        to_Conv("F6", 84, 1, offset="(1.25,0,0)", to="(C5-east)", height=1, depth=84, width=1, caption="FC2"),
        to_connection("C5", "F6"),
        # 输出层: Softmax 分类
        to_SoftMax("O7", 10, offset="(2,0,0)", to="(F6-east)", width=1, height=1, depth=10, caption="Output"),
        to_connection("F6", "O7"),
        to_end()
    ]
    return arch

# 生成pdf格式图片，viz_lenet5_PlotNeuralNet.pdf
def main():
    arch = lenet5_architecture()
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
