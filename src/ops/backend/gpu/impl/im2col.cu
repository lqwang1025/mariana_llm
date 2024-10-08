/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/im2col.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-08:08:19:49
 * Description:
 * 
 */

#include <ops/backend/gpu/impl/im2col.h>

namespace mariana {

__device__ inline bool cuda_is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template<typename T>
__global__ void __im2col_kernel(const T* input, T* out, uint32_t IW, uint32_t IH, uint32_t KW, uint32_t KH, uint32_t IC, uint32_t OW, uint32_t OH, uint32_t dilation_h, uint32_t dilation_w, uint32_t pad_t, uint32_t pad_l, uint32_t stride_h, uint32_t stride_w, uint32_t input_stride_1) {
    int32_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= OH*OW*IC*KW*KH) return;
    int32_t _idx = index;
    const int32_t kw = _idx % KW;
    _idx /= KW;
    const int32_t kh = _idx % KH;
    _idx /= KH;
    const int32_t ic = _idx % IC;
    _idx /= IC;
    const int32_t ow = _idx % OW;
    _idx /= OW;
    const int32_t oh = _idx;
    const int32_t ih = (kh*dilation_h) - pad_t + stride_h*oh;
    const int32_t iw = (kw*dilation_w) - pad_l + stride_w*ow;
    if (!cuda_is_a_ge_zero_and_a_lt_b(ih, IH) ||
        !cuda_is_a_ge_zero_and_a_lt_b(iw, IW) ) { // in padding zone
        out[index] = 0;
    } else {
        const uint32_t iidx = ic*input_stride_1+ih*IW+iw;
        out[index] = input[iidx];
    }
}

void im2col(SchedParam sched_param, const Tensor& input, Tensor& out, int32_t kernel_h, int32_t kernel_w, int32_t pad_t, int32_t pad_l, int32_t pad_b, int32_t pad_r, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, int32_t groups, CUDAContext* cuda_ctx) {
    uint32_t distance = sched_param.this_thread_end_index() - sched_param.this_thread_begin_index();
    if (input.dtype().match<float>()) {
        const uint32_t ioffset = input.stride_at(0);
        const uint32_t ooffset = out.stride_at(0);
        const uint32_t IC      = input.dim_at(1);
        const int32_t  OH      = out.dim_at(1);
        const int32_t  OW      = out.dim_at(2);
        float* input_ptr = input.unsafe_ptr<float>(sched_param.this_thread_begin_index()*ioffset);
        float* out_ptr   = out.unsafe_ptr<float>(sched_param.this_thread_begin_index()*ooffset);
        __im2col_kernel<float><<<get_cuda_gridsize(ooffset, CUDA_IMG2COL_BLOCK_SIZE),
            CUDA_IMG2COL_BLOCK_SIZE, 0, cuda_ctx->stream(sched_param.id_thread)>>>(input_ptr, out_ptr, input.dim_at(3), input.dim_at(2), kernel_w, kernel_h, IC, OW, OH, dilation_h, dilation_w, pad_t, pad_l, stride_h, stride_w, input.stride_at(1));
        cuda_ctx->stream_sync(cuda_ctx->stream(sched_param.id_thread));
    } else {
        MLOG(FATAL)<<"IM2COL unsupport datatype:"<<input.dtype().name();
    }
}

} // namespace mariana
