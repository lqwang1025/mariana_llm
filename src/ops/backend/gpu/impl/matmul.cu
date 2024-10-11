/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/matmul.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-09-26:17:28:12
 * Description:
 * 
 */

#include <ops/backend/gpu/impl/matmul.h>

namespace mariana {

__device__ float __mm_gelu_single_kernel(float x) {
    return 0.5f*x*(1.f+erff(sqrtf(0.5f)*x));
}

__device__ float __mm_relu_single_kernel(float x) {
    return MAX(x, 0.f);
}

template<typename T>
__global__ void __matmul_kernel(const T* input, const T* weight, const T* bias, T* out, uint32_t distance, uint32_t oh, uint32_t ow, uint32_t k, uint32_t bias_size, T alpha, T beta, OpCategory act_cate) {
    int32_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= oh*ow*distance) return;
    uint32_t idx = index;
    const int32_t w_idx = idx % ow;
    idx /= ow;
    const int32_t h_idx = idx % oh;
    idx /= oh;
    const int32_t n_idx = idx;
    T sum = 0;
    for (uint32_t i = 0; i < k; ++i) {
        sum += input[n_idx*k*oh+h_idx*k+i]*weight[w_idx*k+i];
    }
    T _bias = 0;
    if (ow == bias_size) {
        _bias = bias[w_idx];
    } else if (oh == bias_size) {
        _bias = bias[h_idx];
    }
    if (act_cate == OpCategory::GELU) {
        out[index] = __mm_gelu_single_kernel(alpha*sum+beta*_bias);
    } else if (act_cate == OpCategory::RELU) {
        out[index] = __mm_relu_single_kernel(alpha*sum+beta*_bias);
    } else {
        out[index] = alpha*sum+beta*_bias;
    }
}

void matmul(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& out, float alpha, float beta, OpCategory act_cate, CUDAContext* cuda_ctx) {
    if (out.dtype().match<float>()) {
        const uint32_t nok = weight.dim_at(1);
        const uint32_t ioffset = input.stride_at(0);
        const uint32_t ooffset = out.stride_at(0);
        const uint32_t oh = out.dim_at(1);
        const uint32_t ow = out.dim_at(2);
        float* weight_ptr = weight.unsafe_ptr<float>(0);
        float* bias_ptr = nullptr;
        if (bias.total_size() != 0) {
            bias_ptr = bias.unsafe_ptr<float>(0);
        }
        uint32_t distance = sched_param.this_thread_end_index() - sched_param.this_thread_begin_index();
        float* input_ptr = input.unsafe_ptr<float>(sched_param.this_thread_begin_index()*ioffset);
        float* out_ptr = out.unsafe_ptr<float>(sched_param.this_thread_begin_index()*ooffset);
        __matmul_kernel<float><<<get_cuda_gridsize(distance*ooffset, CUDA_MATMUL_BLOCK_SIZE),
            CUDA_MATMUL_BLOCK_SIZE, 0, cuda_ctx->stream(sched_param.id_thread)>>>(input_ptr, weight_ptr, bias_ptr, out_ptr, distance, oh, ow, nok, bias.total_size(), alpha, beta, act_cate);
        cuda_ctx->stream_sync(cuda_ctx->stream(sched_param.id_thread));
    } else {
        MLOG(FATAL)<<"Matmul unsupport datatype:"<<out.dtype().name();
    }
}

} // namespace mariana
