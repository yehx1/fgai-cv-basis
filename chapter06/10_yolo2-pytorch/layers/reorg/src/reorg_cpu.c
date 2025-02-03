#include <torch/extension.h>

int reorg_cpu(torch::Tensor x_tensor, int w, int h, int c, int batch, int stride, int forward, torch::Tensor out_tensor)
{
    // 检查输入是否是连续的（在 PyTorch 中确保效率）
    x_tensor = x_tensor.contiguous();
    out_tensor = out_tensor.contiguous();

    // 获取指向张量数据的指针
    float *x = x_tensor.data_ptr<float>();
    float *out = out_tensor.data_ptr<float>();

    int out_c = c / (stride * stride);
    
    // 循环遍历所有维度，执行 reorg 操作
    for (int b = 0; b < batch; ++b) {
        for (int k = 0; k < c; ++k) {
            for (int j = 0; j < h; ++j) {
                for (int i = 0; i < w; ++i) {
                    int in_index = i + w * (j + h * (k + c * b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i * stride + offset % stride;
                    int h2 = j * stride + offset / stride;
                    int out_index = w2 + w * stride * (h2 + h * stride * (c2 + out_c * b));
                    
                    if (forward) {
                        out[out_index] = x[in_index];
                    } else {
                        out[in_index] = x[out_index];
                    }
                }
            }
        }
    }

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reorg_cpu", &reorg_cpu, "Reorganization layer (CPU)");
}
