# 1. 导入所需的 Keras 模块，包括用于创建模型的 Sequential 和 Dense 层，以及 plot_model 用于生成模型结构图。
import os
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


if __name__ == '__main__':
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    print("当前文件所在目录:", current_dir)
    # 创建结果保存目录
    save_dir = os.path.join(current_dir, 'result')
    os.makedirs(save_dir, exist_ok=True)

    # 2. 定义一个顺序模型，其中第一层是具有 64 个单元的全连接层（激活函数为 relu），第二层是具有 10 个单元的 softmax 输出层。
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(100,)))
    model.add(Dense(10, activation='softmax'))

    # 3. 生成并保存模型结构图到result/viz_keras_plot_model.png文件，并显示每层的输出形状和层名称，图的方向为自上而下（TB）。
    plot_model(model, to_file=f'{save_dir}/viz_keras_plot_model.png', show_shapes=True, show_layer_names=True, rankdir='TB')
