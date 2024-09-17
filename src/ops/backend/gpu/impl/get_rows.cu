/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/get_rows.cu
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-14:09:49:07
 * Description:
 * 
 */
#include <cstdio>
#include <ops/backend/gpu/impl/get_rows.h>
#include <core/backend/gpu/cuda_allocator.h>

namespace mariana {

template<typename T>
__global__ void __get_rows_kernel(const int32_t* idx_ptr, T* dst, int32_t ne) {
    int32_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    printf("d:%d\n",index);
}

void get_rows(SchedParam sched_param, const Tensor& indeices, const Tensor& embedding, Tensor& out, CUDAContext* cuda_ctx) {
    MLOG(INFO)<<cuda_ctx->name<<" "<<device_string(indeices.device());
    const int32_t nr   = indeices.dim_at(1); // token_size
    const int32_t ne   = embedding.dim_at(1);
    IAllocator* allocator = get_allocator(DataOn::GPU);
    CudaIAllocator* cuda_alloc = static_cast<CudaIAllocator*>(allocator);
    uint32_t distance = sched_param.this_thread_end_index() - sched_param.this_thread_begin_index();
    MLOG(INFO)<<distance<<" "<<ne;
     __get_rows_kernel<float><<<get_cuda_gridsize(distance, CUDA_GET_ROWS_BLOCK_SIZE),
        CUDA_GET_ROWS_BLOCK_SIZE, 0, cuda_ctx->stream()>>>(ne);
}

} // namespace mariana
