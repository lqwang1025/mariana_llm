/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/softmax.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-10:05:50:17
 * Description:
 * 
 */

#include <cfloat>

#include <ops/backend/gpu/impl/softmax.h>
#include <core/backend/gpu/cuda_allocator.h>

namespace mariana {

__global__ void __softmax3_fp32_kernel(const float* input_ptr, float* out_ptr, uint32_t distance, uint32_t H, uint32_t W) {
    int32_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= distance*H) return;
    uint32_t idx = index;
    const int32_t h_idx = idx % H;
    idx /= H;
    const int32_t d_idx = idx;
    float maxval = -FLT_MAX;
    for (uint32_t w = 0; w < W; ++w) {
        float val = input_ptr[d_idx*H*W+h_idx*W+w];
        if (val > maxval) {
            maxval = val;
        }
    }
    float expsum = 0.f;
    for (uint32_t w = 0; w < W; ++w) {
        float val = input_ptr[d_idx*H*W+h_idx*W+w];
        float expv = expf(val-maxval);
        expsum += expv;
        out_ptr[d_idx*H*W+h_idx*W+w] = expv;
    }
    float expsum_inv = expsum == 0.f ? 0.f : 1.f/expsum;
    for (uint32_t w = 0; w < W; ++w) {
        out_ptr[d_idx*H*W+h_idx*W+w] *= expsum_inv;
    }
}

void softmax3(SchedParam sched_param, const Tensor& input, Tensor& out, int32_t dim, CUDAContext* cuda_ctx) {
    cuda_set_device(cuda_ctx->device);
    if (out.dtype().match<float>()) {
        if (dim != -1) {
            MLOG(FATAL)<<"softmax unsupport dim:"<<dim;
        }
        uint32_t distance = sched_param.this_thread_end_index() - sched_param.this_thread_begin_index();
        uint32_t H = out.dim_at(1);
        uint32_t W = out.dim_at(2);
        const uint32_t ioffset = input.stride_at(0);
        const uint32_t ooffset = out.stride_at(0);
        float* input_ptr = input.unsafe_ptr<float>(sched_param.this_thread_begin_index()*ioffset);
        float* out_ptr = out.unsafe_ptr<float>(sched_param.this_thread_begin_index()*ooffset);
        __softmax3_fp32_kernel<<<get_cuda_gridsize(distance*H, CUDA_SOFTMAX_BLOCK_SIZE),
            CUDA_SOFTMAX_BLOCK_SIZE, 0, cuda_ctx->stream(sched_param.id_thread)>>>(input_ptr, out_ptr, distance, H, W);
        cuda_ctx->stream_sync(cuda_ctx->stream(sched_param.id_thread));
    }
}

} // namespace mariana
