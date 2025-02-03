import os
import onnx
import onnxoptimizer

if __name__ == '__main__':
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    # 获取当前文件所在的目录
    current_dir = os.path.dirname(current_file_path)
    print("当前文件所在目录:", current_dir)
    # 创建结果保存目录
    save_dir = os.path.join(current_dir, 'result')
    os.makedirs(save_dir, exist_ok=True)

    # 1) 从文件加载原始模型
    model_path = f"{save_dir}/resnet18.onnx"
    original_model = onnx.load(model_path)
    onnx.checker.check_model(original_model)  # 原模型检验

    # 2) 定义需要应用的优化 pass
    # 这里仅列举一些常见的，可根据需要添加或删除
    passes = [
        "eliminate_identity",         # 移除多余的 Identity 算子
        # "eliminate_nop_transpose",    # 移除无效的 Transpose 算子
        # "eliminate_nop_pad",          # 移除无效的 Pad 算子
        # "fuse_consecutive_transposes",# 合并连续的 Transpose
        # "fuse_add_bias_into_conv",    # 将 Add Bias 融合进 Conv 算子
        # "eliminate_deadend",          # 移除没有输出的节点
        # "eliminate_dead_graph_input"  # 移除未使用的图输入
    ]

    # 3) 调用 onnxoptimizer.optimize(model, passes) 获取优化后的模型
    optimized_model = onnxoptimizer.optimize(original_model, passes)

    # 4) 保存优化后的模型
    optimized_model_path = f"{save_dir}/resnet18_optimized.onnx"
    onnx.save(optimized_model, optimized_model_path)
    print(f"Optimized model saved to {optimized_model_path}")