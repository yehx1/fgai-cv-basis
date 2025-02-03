import numpy as np
import onnxruntime as ort

if __name__ == '__main__':
    # 创建推理会话
    session = ort.InferenceSession("./result/resnet18.onnx")
    # 获取输入输出信息
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print("input_name, output_name: ", input_name, output_name)
    # 构造输入
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    # 推理
    result = session.run([output_name], {input_name: dummy_input})
    print("result[0].shape: ", result[0].shape)