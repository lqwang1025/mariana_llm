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

#include <ops/backend/gpu/impl/get_rows.h>
#include <core/backend/gpu/cuda_allocator.h>

namespace mariana {

template<typename T>
__global__ void __get_rows_kernel() {
    
}

void get_rows(SchedParam sched_param, const Tensor& indeices, const Tensor& embedding, Tensor& out, CUDAContext* cuda_ctx) {
    MLOG(INFO)<<cuda_ctx->name<<" "<<device_string(indeices.device());
    const int32_t nr   = indeices.dim_at(1); // token_size
    const int32_t ne   = embedding.dim_at(1);
    IAllocator* allocator = get_allocator(out.device());
    CudaIAllocator* cuda_alloc = static_cast<CudaIAllocator*>(allocator);
    //__get_rows_kernel()<<<>>>();
}

} // namespace mariana
