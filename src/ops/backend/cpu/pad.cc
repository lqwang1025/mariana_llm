/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/pad.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-01:09:07:25
 * Description:
 * 
 */

#include <core/impl/allocator.h>
#include <ops/backend/cpu/pad.h>

namespace mariana {

[[maybe_unused]]static void naive_nchw_pad_hw_fp32(SchedParam sched_param, const Tensor& input, Tensor& out, uint32_t padding[4], float pad_value) {
    uint32_t pad_top    = padding[0];
    uint32_t pad_left   = padding[1];
    uint32_t pad_bottom = padding[2];

    const int32_t H = out.dim_at(2);
    const int32_t W = out.dim_at(3);

    // split chunk by NxC
    const int32_t ostride_plane = out.stride_at(1);
    const int32_t istride_plane = input.stride_at(1);
    const int32_t iw            = input.dim_at(3);
    
    IAllocator* allocator = get_allocator(out.device());
    for (uint32_t row = sched_param.this_thread_begin_index(); row < sched_param.this_thread_end_index(); ++row) {
        float* out_ptr = out.unsafe_ptr<float>(row*ostride_plane);
        float* in_ptr  = input.unsafe_ptr<float>(row*istride_plane);
        uint32_t h = 0;
        for (; h < pad_top; ++h) {
            for (int32_t w = 0; w < W; ++w) {
                out_ptr[h*W+w] = pad_value;
            }
        }
        
        for (; h < (H-pad_bottom); ++h) {
            uint32_t w = 0;
            for (; w < pad_left; ++w) {
                out_ptr[h*W+w] = pad_value;
            }
            allocator->memcpy(out_ptr+h*W+pad_left, in_ptr+(h-pad_top)*iw, sizeof(float)*iw);
            for (w = pad_left+iw; w < (uint32_t)W; ++w) {
                out_ptr[h*W+w] = pad_value;
            }
        }
        
        for (; h < (uint32_t)H; ++h) {
            for (int32_t w = 0; w < W; ++w) {
                out_ptr[h*W+w] = pad_value;
            }
        }
    }
}


// (padding_left,  padding_right,
//  padding_top,   padding_bottom,
//  padding_front, padding_back )
static void naive_nchw_pad_fp32(SchedParam sched_param, const Tensor& input, Tensor& out, uint32_t padding[6], float pad_value) {
    uint32_t pl  = padding[0];
    uint32_t pr  = padding[1];
    uint32_t pt  = padding[2];
    uint32_t pb  = padding[3];
    uint32_t pf  = padding[4];
    uint32_t pbk = padding[5];

    const int32_t C = out.dim_at(1);
    const int32_t H = out.dim_at(2);
    const int32_t W = out.dim_at(3);

    for (uint32_t idx = sched_param.this_thread_begin_index(); idx < sched_param.this_thread_end_index(); ++idx) { // row = n*h*w
        uint32_t _idx = idx;
        uint32_t w = _idx%W;
        _idx /= W;
        uint32_t h = _idx%H;
        _idx /= H;
        uint32_t c = _idx%C;
        _idx /= C;
        uint32_t n = _idx;
        uint32_t oidx = n*out.stride_at(0)+c*out.stride_at(1)
            +h*out.stride_at(2)+w;
        float* out_ptr = out.unsafe_ptr<float>(oidx);
        if (w < pl || (W-pr-1) < w) {
            out_ptr[0] = pad_value;
            continue;
        }
        if (h < pt || (H-pb-1) < h) {
            out_ptr[0] = pad_value;
            continue;
        }
        if (c < pf || (C-pbk-1) < c) {
            out_ptr[0] = pad_value;
            continue;
        }
        
        uint32_t ic = c-pf;
        uint32_t ih = h-pt;
        uint32_t iw = w-pl;
        
        uint32_t iidx = n*input.stride_at(0)+ic*input.stride_at(1)
            +ih*input.stride_at(2)+iw;
        
        float* in_ptr  = input.unsafe_ptr<float>(iidx);
        out_ptr[0] = in_ptr[0];
    }
}

void nchw_pad(SchedParam sched_param, const Tensor& input, Tensor& out, uint32_t padding[6], float pad_value) {
    naive_nchw_pad_fp32(sched_param, input, out, padding, pad_value);
}

} // namespace mariana

