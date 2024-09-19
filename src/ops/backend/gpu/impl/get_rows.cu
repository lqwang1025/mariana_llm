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

#include <utils/mariana_define.h>
#include <ops/backend/gpu/impl/get_rows.h>

namespace mariana {

template<typename T>
__global__ void __get_rows_kernel(const int32_t* idx_ptr, const T* embedding, T* dst, int32_t batch_size, int32_t token_size, int32_t ne, int32_t begin) {
    int32_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    index += begin;
    if (index >= batch_size*token_size) return;
    const int32_t t = index%token_size;
    const int32_t b = index/token_size;
    int32_t point = idx_ptr[index-begin];
    for (int32_t i = 0; i < ne; ++i) {
        dst[b*token_size*ne+t*ne+i] = embedding[point*ne+i];
    }
}

void get_rows(SchedParam sched_param, const Tensor& indeices, const Tensor& embedding, Tensor& out, CUDAContext* cuda_ctx) {
    const int32_t bs   = indeices.dim_at(0); // token_size
    const int32_t nr   = indeices.dim_at(1); // token_size
    const int32_t ne   = embedding.dim_at(1);
    uint32_t distance = sched_param.this_thread_end_index() - sched_param.this_thread_begin_index();
    MLOG_IF(FATAL, indeices.dtype().match<int32_t>()==false)<<"Get Rows operator support data type int32 input only.";
    if (embedding.dtype().match<float>()) {
        float* weight   = embedding.unsafe_ptr<float>(0);
        int32_t* index  = indeices.unsafe_ptr<int32_t>(sched_param.this_thread_begin_index());
        float* dst      = out.unsafe_ptr<float>(0);
        __get_rows_kernel<float><<<get_cuda_gridsize(distance, CUDA_GET_ROWS_BLOCK_SIZE),
            CUDA_GET_ROWS_BLOCK_SIZE, 0, cuda_ctx->stream(sched_param.id_thread)>>>(index, weight, dst, bs, nr, ne, sched_param.this_thread_begin_index());
        cuda_ctx->stream_sync(cuda_ctx->stream(sched_param.id_thread));
    } else {
        MLOG(FATAL)<<"Get row unsupport datatype:"<<embedding.dtype().name();
    }
}

} // namespace mariana
