/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/layer_norm.cu
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-21:07:55:44
 * Description:
 * 
 */

#include <ops/backend/gpu/impl/layer_norm.h>

namespace mariana {

template<typename T>
__global__ void __layer_normlization_kernel(const T* input_ptr, const T* weight, const T* bias, T* out, int32_t nbs) {
    int32_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= nbs) return;
    T mean = 0;
    for (int32_t col = 0; col < nbs; ++col) {
        mean += input_ptr[index];
    }
}
    
void layer_normlization(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, const NormParam& norm_param, Tensor& out, CUDAContext* cuda_ctx) {
    if (out.dtype().match<float>()) {
        const int32_t nbs = input.stride_at(1);
        float* input_ptr  = input.unsafe_ptr<float>(sched_param.this_thread_begin_index()*nbs);
        float* weight_ptr = weight.unsafe_ptr<float>(0);
        float* bias_ptr   = bias.unsafe_ptr<float>(0);
        float* dst_ptr    = out.unsafe_ptr<float>(sched_param.this_thread_begin_index()*nbs);
        __layer_normlization_kernel<float><<<get_cuda_gridsize(nbs, CUDA_LN_BLOCK_SIZE),
            CUDA_LN_BLOCK_SIZE, 0, cuda_ctx->stream(sched_param.id_thread)>>>(input_ptr, weight_ptr, bias_ptr, dst_ptr, nbs);
        cuda_ctx->stream_sync(cuda_ctx->stream(sched_param.id_thread));
    } else {
        MLOG(FATAL)<<"layer norm unsupport datatype:"<<out.dtype().name();
    }
}

} // namespace mariana
