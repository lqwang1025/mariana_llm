/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/pad.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-09:09:57:46
 * Description:
 * 
 */

#include <ops/backend/gpu/impl/pad.h>

namespace mariana {

template<typename T>
__global__ void __nchw_pad_kernel(const T* input, T* out, int32_t distance, int32_t OC, int32_t OH, int32_t OW, T pad_value, uint32_t input_stride_0, uint32_t input_stride_1, uint32_t input_stride_2, uint32_t out_stride_0, uint32_t out_stride_1, uint32_t out_stride_2, uint32_t pl, uint32_t pr, uint32_t pt, uint32_t pb, uint32_t pf, uint32_t pbk) {
    int32_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= distance*OC*OW*OH) return;
    uint32_t _idx = index;
    uint32_t w = _idx%OW;
    _idx /= OW;
    uint32_t h = _idx%OH;
    _idx /= OH;
    uint32_t c = _idx %OC;
    _idx /= OC;
    uint32_t n = _idx;
    uint32_t oidx = n*out_stride_0+c*out_stride_1+h*out_stride_2+w;
    if (w < pl || (OW-pr-1) < w) {
        out[oidx] = pad_value;
        return;
    }
    if (h < pt || (OH-pb-1) < h) {
        out[oidx] = pad_value;
        return;
    }
    if (c < pf || (OC-pbk-1) < c) {
        out[oidx] = pad_value;
        return;
    }
    uint32_t ic = c-pf;
    uint32_t ih = h-pt;
    uint32_t iw = w-pl;
    uint32_t iidx = n*input_stride_0+ic*input_stride_1+ih*input_stride_2+iw;
    out[oidx] = input[iidx];
}

void nchw_pad(SchedParam sched_param, const Tensor& input, Tensor& out, uint32_t padding[6], float pad_value, CUDAContext* cuda_ctx) {
    if (out.dtype().match<float>()) {
        uint32_t pl  = padding[0];
        uint32_t pr  = padding[1];
        uint32_t pt  = padding[2];
        uint32_t pb  = padding[3];
        uint32_t pf  = padding[4];
        uint32_t pbk = padding[5];
        uint32_t istride_0 = input.stride_at(0);
        uint32_t istride_1 = input.stride_at(1);
        uint32_t istride_2 = input.stride_at(2);
        uint32_t ostride_0 = out.stride_at(0);
        uint32_t ostride_1 = out.stride_at(1);
        uint32_t ostride_2 = out.stride_at(2);
        const int32_t OC = out.dim_at(1);
        const int32_t OH = out.dim_at(2);
        const int32_t OW = out.dim_at(3);
        const uint32_t ioffset = input.stride_at(0);
        const uint32_t ooffset = out.stride_at(0);
        uint32_t distance = sched_param.this_thread_end_index() - sched_param.this_thread_begin_index();
        float* input_ptr = input.unsafe_ptr<float>(sched_param.this_thread_begin_index()*ioffset);
        float* out_ptr = out.unsafe_ptr<float>(sched_param.this_thread_begin_index()*ooffset);
        __nchw_pad_kernel<float><<<get_cuda_gridsize(distance*ooffset, CUDA_PAD_BLOCK_SIZE),
            CUDA_PAD_BLOCK_SIZE, 0, cuda_ctx->stream(sched_param.id_thread)>>>(input_ptr, out_ptr, distance, OC, OH, OW, pad_value, istride_0, istride_1, istride_2, ostride_0, ostride_1, ostride_2, pl, pr, pt, pb, pf, pbk);
        cuda_ctx->stream_sync(cuda_ctx->stream(sched_param.id_thread));
    } else {
        MLOG(FATAL)<<"nchw_pad unsupport datatype:"<<out.dtype().name();
    }
}

} // namespace mariana 
