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

#include <ops/layer_norm.h>
#include <ops/backend/gpu/impl/layer_norm.h>

namespace mariana {

template<typename T>
__global__ void __layer_normlization_kernel(const T* input_ptr, const T* weight, const T* bias, T* out, float epsilon, int32_t distance, int32_t c, int32_t l) {
    int32_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= c*distance) return;
    T mean = 0;
    int32_t offset = index*l;
    for (int32_t col = 0; col < l; ++col) {
        mean += input_ptr[offset+col];
    }
    mean /= l;
    T var = 0;
    for (int32_t col = 0; col < l; ++col) {
        T x_shift = input_ptr[offset+col]-mean;
        var += pow(x_shift, 2);
    }
    var /= l;
    var = 1.f/sqrtf(var + epsilon);
    for (int32_t col = 0; col < l; ++col) {
        T val = input_ptr[offset+col];
        T n = (var*(val - mean));
        T o = n*weight[col] + bias[col];
        out[offset+col] = o;
    }
}
    
void layer_normlization(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, const NormParam& norm_param, Tensor& out, CUDAContext* cuda_ctx) {
    if (out.dtype().match<float>()) {
        const int32_t c = input.dim_at(1);
        const int32_t l = input.dim_at(2);
        uint32_t distance = sched_param.this_thread_end_index() - sched_param.this_thread_begin_index();
        float* input_ptr  = input.unsafe_ptr<float>(sched_param.this_thread_begin_index()*c*l);
        float* weight_ptr = weight.unsafe_ptr<float>(0);
        float* bias_ptr   = bias.unsafe_ptr<float>(0);
        float* dst_ptr    = out.unsafe_ptr<float>(sched_param.this_thread_begin_index()*c*l);
        __layer_normlization_kernel<float><<<get_cuda_gridsize(distance*c, CUDA_LN_BLOCK_SIZE),
            CUDA_LN_BLOCK_SIZE, 0, cuda_ctx->stream(sched_param.id_thread)>>>(input_ptr, weight_ptr, bias_ptr, dst_ptr, norm_param.epsilon, distance, c, l);
        cuda_ctx->stream_sync(cuda_ctx->stream(sched_param.id_thread));
    } else {
        MLOG(FATAL)<<"layer norm unsupport datatype:"<<out.dtype().name();
    }
}

template<typename T>
__global__ void __group_normlization_kernel(const T* input_ptr, const T* weight, const T* bias, T* out, float epsilon, int32_t distance, int32_t c, int32_t l, int32_t GC, int32_t HW) {
    int32_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= c*distance) return;
    T mean = 0;
    int32_t offset = index*l;
    for (int32_t col = 0; col < l; ++col) {
        mean += input_ptr[offset+col];
    }
    mean /= l;
    T var = 0;
    for (int32_t col = 0; col < l; ++col) {
        T x_shift = input_ptr[offset+col]-mean;
        var += pow(x_shift, 2);
    }
    var /= l;
    var = 1.f/sqrtf(var + epsilon);

    const int32_t g = index%c;
    for (int32_t col = 0; col < l; ++col) {
        const int32_t gc = (col/HW) % GC;
        T val = input_ptr[offset+col];
        T n = (var*(val - mean));
        T o = n*weight[g*GC+gc] + bias[g*GC+gc];
        out[offset+col] = o;
    }
}

void group_normlization(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& out, NormParam& norm_param, CUDAContext* cuda_ctx) {
    if (out.dtype().match<float>()) {
        const int32_t c = input.dim_at(1);
        const int32_t nbs = input.stride_at(1);
        const int32_t GC  = input.dim_at(2);
        const int32_t HW  = input.dim_at(3);
        uint32_t distance = sched_param.this_thread_end_index() - sched_param.this_thread_begin_index();
        float* input_ptr  = input.unsafe_ptr<float>(sched_param.this_thread_begin_index()*input.stride_at(0));
        float* weight_ptr = weight.unsafe_ptr<float>(0);
        float* bias_ptr   = bias.unsafe_ptr<float>(0);
        float* dst_ptr    = out.unsafe_ptr<float>(sched_param.this_thread_begin_index()*out.stride_at(0));
        __group_normlization_kernel<float><<<get_cuda_gridsize(distance*c, CUDA_LN_BLOCK_SIZE),
            CUDA_LN_BLOCK_SIZE, 0, cuda_ctx->stream(sched_param.id_thread)>>>(input_ptr, weight_ptr, bias_ptr, dst_ptr, norm_param.epsilon, distance, c, nbs, GC, HW);
        cuda_ctx->stream_sync(cuda_ctx->stream(sched_param.id_thread));
    } else {
        MLOG(FATAL)<<"group_normlization unsupport datatype:"<<out.dtype().name();
    }
}

} // namespace mariana
