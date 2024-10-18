/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/grounding_dino_sine_position_embedding.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-15:10:01:35
 * Description:
 * 
 */

#include <ops/backend/gpu/impl/grounding_dino_sine_position_embedding.h>

namespace mariana {

template<typename T>
__global__ void __gdspe_kernel(const T* input, T* out, T scale, T temperature, int32_t o_dim_0, int32_t o_dim_1, int32_t o_dim_2, int32_t o_dim_3) {
    int32_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= o_dim_0*o_dim_1*o_dim_2*o_dim_3) return;
    uint32_t embedding_dim = (uint32_t)o_dim_1/2;
    uint32_t H = (uint32_t)o_dim_2;
    uint32_t W = (uint32_t)o_dim_3;
    const float eps = 1e-6;
    uint32_t idx = index;
    uint32_t w = idx % o_dim_3;
    idx /= o_dim_3;
    uint32_t h = idx % o_dim_2;
    idx /= o_dim_2;
    uint32_t c = idx % o_dim_1;
    
    if (c < embedding_dim) { // pos_y
        T dim_t = c;
        T factor = pow(temperature, 2*floor(dim_t/2)/embedding_dim);
        T val = (h+1)/(H+eps) * scale;
        // (h+1)/(H+eps) * scale;
        if (c%2 == 0) {
            out[index] = sin(val / factor);
        } else {
            out[index] = cos(val / factor);
        }
    } else { // pos_x
        float dim_t  = c-embedding_dim;
        float factor = pow(temperature, 2*floor(dim_t/2)/embedding_dim);
        float val = (w+1)/(W+eps) * scale;
        // (h+1)/(H+eps) * scale;
        if (c%2 == 0) {
            out[index] = sin(val / factor);
        } else {
            out[index] = cos(val / factor);
        }
    }
}

void grounding_dino_sine_position_embedding(SchedParam sched_param, const Tensor& input, Tensor& out, float scale, float temperature, CUDAContext* cuda_ctx) {
    if (out.dtype().match<float>()) {
        const int32_t dim0 = out.dim_at(0);
        const int32_t dim1 = out.dim_at(1);
        const int32_t dim2 = out.dim_at(2);
        const int32_t dim3 = out.dim_at(3);
        float* input_ptr = input.unsafe_ptr<float>(0);
        float* out_ptr = out.unsafe_ptr<float>(0);
        __gdspe_kernel<float><<<get_cuda_gridsize(out.total_size(), CUDA_GDSPE_BLOCK_SIZE),
            CUDA_GDSPE_BLOCK_SIZE, 0, cuda_ctx->stream(sched_param.id_thread)>>>(input_ptr, out_ptr, scale, temperature, dim0, dim1, dim2, dim3);
        cuda_ctx->stream_sync(cuda_ctx->stream(sched_param.id_thread));
    } else {
        MLOG(FATAL)<<"grounding_dino_sine_position_embedding unsupport datatype:"<<out.dtype().name();
    }
}

} // namespace mariana
