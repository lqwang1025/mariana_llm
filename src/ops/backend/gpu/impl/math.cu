/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/math.cu
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-19:05:33:32
 * Description:
 * 
 */

#include <utils/mariana_define.h>
#include <ops/backend/gpu/impl/math.h>

namespace mariana {

template<typename T>
__global__ void __add_ele_kernel(const T* a_ptr, const T* b_ptr, T* dst, uint32_t total_size, uint32_t begin) {
    int32_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index+begin >= total_size) return;
    dst[index] = a_ptr[index] + b_ptr[index];
}

void add_ele(SchedParam sched_param, const Tensor& a, const Tensor& b, Tensor& out, CUDAContext* cuda_ctx) {
    if (a.dtype().match<float>()) {
        float* a_ptr = a.unsafe_ptr<float>(sched_param.this_thread_begin_index());
        float* b_ptr = b.unsafe_ptr<float>(sched_param.this_thread_begin_index());
        float* c_ptr = out.unsafe_ptr<float>(sched_param.this_thread_begin_index());
        uint32_t distance = sched_param.this_thread_end_index() - sched_param.this_thread_begin_index();
        __add_ele_kernel<float><<<get_cuda_gridsize(distance, CUDA_ADD_BLOCK_SIZE),
            CUDA_ADD_BLOCK_SIZE, 0, cuda_ctx->stream(sched_param.id_thread)>>>(a_ptr, b_ptr, c_ptr, a.total_size(), sched_param.this_thread_begin_index());
        cuda_ctx->stream_sync(cuda_ctx->stream(sched_param.id_thread));
    } else {
        MLOG(FATAL)<<"Add unsupport datatype:"<<a.dtype().name();
    }
}

} // namespace mariana
