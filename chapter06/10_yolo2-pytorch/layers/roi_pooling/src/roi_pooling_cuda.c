#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda/roi_pooling_kernel.h"

int roi_pooling_forward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                             torch::Tensor features, torch::Tensor rois, torch::Tensor output, torch::Tensor argmax)
{
    // 确保输入张量在 CUDA 上，并且是连续的，以便于高效计算
    features = features.contiguous();
    rois = rois.contiguous();
    output = output.contiguous();
    argmax = argmax.contiguous();

    // 获取张量数据指针
    float* data_flat = features.data_ptr<float>();
    float* rois_flat = rois.data_ptr<float>();
    float* output_flat = output.data_ptr<float>();
    int* argmax_flat = argmax.data_ptr<int>();

    // 获取张量的维度
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5) {
        return 0;
    }

    int batch_size = features.size(0);
    if (batch_size != 1) {
        return 0;
    }
    int data_height = features.size(2);
    int data_width = features.size(3);
    int num_channels = features.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    ROIPoolForwardLaucher(
        data_flat, spatial_scale, num_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, rois_flat,
        output_flat, argmax_flat, stream);

    return 1;
}

int roi_pooling_backward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                              torch::Tensor top_grad, torch::Tensor rois, torch::Tensor bottom_grad, torch::Tensor argmax)
{
    // 确保输入张量在 CUDA 上，并且是连续的，以便于高效计算
    top_grad = top_grad.contiguous();
    rois = rois.contiguous();
    bottom_grad = bottom_grad.contiguous();
    argmax = argmax.contiguous();

    // 获取张量数据指针
    float* top_grad_flat = top_grad.data_ptr<float>();
    float* rois_flat = rois.data_ptr<float>();
    float* bottom_grad_flat = bottom_grad.data_ptr<float>();
    int* argmax_flat = argmax.data_ptr<int>();

    // 获取张量的维度
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5) {
        return 0;
    }

    int batch_size = bottom_grad.size(0);
    if (batch_size != 1) {
        return 0;
    }
    int data_height = bottom_grad.size(2);
    int data_width = bottom_grad.size(3);
    int num_channels = bottom_grad.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    ROIPoolBackwardLaucher(
        top_grad_flat, spatial_scale, batch_size, num_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, rois_flat,
        bottom_grad_flat, argmax_flat, stream);

    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("roi_pooling_forward_cuda", &roi_pooling_forward_cuda, "ROI Pooling forward (CUDA)");
    m.def("roi_pooling_backward_cuda", &roi_pooling_backward_cuda, "ROI Pooling backward (CUDA)");
}
