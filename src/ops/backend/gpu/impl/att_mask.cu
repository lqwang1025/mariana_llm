/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/att_mask.cu
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-18:17:10:26
 * Description:
 * 
 */

#include <cfloat>

#include <ops/backend/gpu/impl/att_mask.h>

namespace mariana {

template<typename T>
__global__ void __att_mask_cast_to_kernel(const uint8_t* input, T* dst, uint32_t total_size, uint32_t begin) {
    int32_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index+begin >= total_size) return;
    uint8_t item = input[index];
    dst[index] = item == 0 ? -FLT_MAX : 0;
}
    
void att_mask_cast_to(SchedParam sched_param, const Tensor& input, Tensor& out, CUDAContext* cuda_ctx) {
    MLOG_IF(FATAL, input.dtype().match<uint8_t>()==false)<<"att_mask operator support data type uint8 input only.";
    uint32_t distance = sched_param.this_thread_end_index() - sched_param.this_thread_begin_index();
    if (out.dtype().match<float>()) {
        uint8_t* input_ptr  = input.unsafe_ptr<uint8_t>(sched_param.this_thread_begin_index());
        float* dst_ptr      = out.unsafe_ptr<float>(sched_param.this_thread_begin_index());
        __att_mask_cast_to_kernel<float><<<get_cuda_gridsize(distance, CUDA_ATT_MASK_CAST_BLOCK_SIZE),
            CUDA_ATT_MASK_CAST_BLOCK_SIZE, 0, cuda_ctx->stream(sched_param.id_thread)>>>(input_ptr, dst_ptr, input.total_size(), sched_param.this_thread_begin_index());
        cuda_ctx->stream_sync(cuda_ctx->stream(sched_param.id_thread));
    } else {
        MLOG(FATAL)<<"att_mask_cast_to unsupport datatype:"<<out.dtype().name();
    }
}

} // namespace mariana
