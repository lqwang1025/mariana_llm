/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/roll.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-09:10:46:19
 * Description:
 * 
 */

#include <ops/backend/gpu/impl/roll.h>

namespace mariana {

template<typename T>
__global__ void __roll4_kernel(const T* input, T* out, int32_t OC, int32_t OH, int32_t OW, T pad_value, uint32_t input_stride_1, uint32_t input_stride_2, uint32_t out_stride_1, uint32_t out_stride_2, ) {
    
}

void roll4(SchedParam sched_param, const Tensor& input, Tensor& out, const RollParam& param, CUDAContext* cuda_ctx) {
    if (out.dtype().match<float>()) {
        uint32_t istride_1 = input.stride_at(1);
        uint32_t ostride_1 = out.stride_at(1);
        uint32_t istride_2 = input.stride_at(2);
        uint32_t ostride_2 = out.stride_at(2);
        const int32_t OC = out.dim_at(1);
        const int32_t OH = out.dim_at(2);
        const int32_t OW = out.dim_at(3);
        const uint32_t ioffset = input.stride_at(0);
        const uint32_t ooffset = out.stride_at(0);
        float* input_ptr = input.unsafe_ptr<float>(sched_param.this_thread_begin_index()*ioffset);
        float* out_ptr = out.unsafe_ptr<float>(sched_param.this_thread_begin_index()*ooffset);
        __roll4_kernel<float><<<get_cuda_gridsize(ooffset, CUDA_ROLL_BLOCK_SIZE),
            CUDA_ROLL_BLOCK_SIZE, 0, cuda_ctx->stream(sched_param.id_thread)>>>(input_ptr, out_ptr, OC, OH, OW, pad_value, istride_1, istride_2, ostride_1, ostride_2);
        cuda_ctx->stream_sync(cuda_ctx->stream(sched_param.id_thread));
    } else {
        MLOG(FATAL)<<"roll4 unsupport datatype:"<<out.dtype().name();
    }

}

} // namespace mariana
