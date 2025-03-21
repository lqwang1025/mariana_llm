/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/rope.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-09:10:46:19
 * Description:
 * 
 */

#include <ops/backend/gpu/impl/rope.h>

namespace mariana {

__global__ void __rope_fp32_kernel(const int32_t* pos_ids, const float* inv_freq, float* sin, float* cos, float attn_scaling, int32_t odim0, int32_t odim1, int32_t odim2) {
    int32_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= odim0*odim1*odim2) return;
    int32_t idx = index;
    int32_t idx2 = idx % odim2;
    idx /= odim2;
    int32_t idx1 = idx % odim1;
    idx /= odim1;
    int32_t idx0 = idx;
    int32_t offset = idx2 % (odim2/2);
    float freq = static_cast<float>(pos_ids[idx1]) * inv_freq[offset];
    int32_t oindex = idx0*odim1*odim2 + idx1*odim2 + idx2;
    sin[oindex] = sinf(freq) * attn_scaling;
    cos[oindex] = cosf(freq) * attn_scaling;
}

void rope(SchedParam sched_param, const Tensor& input, Tensor& sin, Tensor& cos, const ROPEParam& param, CUDAContext* cuda_ctx) {
    cuda_set_device(cuda_ctx->device);
    if (sin.dtype().match<float>()) { // input is postion_ids
        uint32_t istride_0 = input.stride_at(0);
        uint32_t distance = sched_param.this_thread_end_index() - sched_param.this_thread_begin_index();
        int32_t* pos_ids_ptr = input.unsafe_ptr<int32_t>(sched_param.this_thread_begin_index()*istride_0);
        uint32_t sin_stride_0 = sin.stride_at(0);
        uint32_t cos_stride_0 = cos.stride_at(0);
        float* sin_ptr = sin.unsafe_ptr<float>(sched_param.this_thread_begin_index()*sin_stride_0);
        float* cos_ptr = cos.unsafe_ptr<float>(sched_param.this_thread_begin_index()*cos_stride_0);
        float* inv_freq_ptr = param.inv_freq.unsafe_ptr<float>(0);
        const int32_t dim1 = sin.dim_at(1);
        const int32_t dim2 = sin.dim_at(2);
        __rope_fp32_kernel<<<get_cuda_gridsize(distance*sin_stride_0, CUDA_ROPE_BLOCK_SIZE), CUDA_ROPE_BLOCK_SIZE,
            0, cuda_ctx->stream(sched_param.id_thread)>>>
            (pos_ids_ptr, inv_freq_ptr, sin_ptr, cos_ptr, param.attention_factor, distance, dim1, dim2);
        cuda_ctx->stream_sync(cuda_ctx->stream(sched_param.id_thread));
    } else {
        MLOG(FATAL)<<"rope unsupport datatype:"<<sin.dtype().name();
    }
}

} // namespace mariana
