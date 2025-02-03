import onnx
import onnxruntime as ort
import numpy as np

if __name__ == "__main__":
    # 加载并检查 ONNX 模型
    model = onnx.load("./result/simple_cnn.onnx")
    onnx.checker.check_model(model)
    print("ONNX Model Check Passed!")

    # 推理
    session = ort.InferenceSession("./result/simple_cnn.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    test_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
    result = session.run([output_name], {input_name: test_input})
    print("Inference Output:", result)