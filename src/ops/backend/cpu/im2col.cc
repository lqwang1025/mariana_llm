/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/im2col.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-26:15:32:38
 * Description:
 * 
 */

#include <ops/backend/cpu/im2col.h>

#include <utils/mariana_define.h>

namespace mariana {

// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
inline static bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// input: [N, IC, IH, IW]
// out: [N, OH, OW, IC*KH*KW]
void im2col_element_split(SchedParam sched_param, const Tensor& input, Tensor& out, int32_t kernel_h, int32_t kernel_w, int32_t pad_t, int32_t pad_l, int32_t pad_b, int32_t pad_r, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, int32_t groups) {
    const int32_t OH = out.dim_at(1);
    const int32_t OW = out.dim_at(2);
    
    const int32_t IC = input.dim_at(1);

    const int32_t IH = input.dim_at(2);
    const int32_t IW = input.dim_at(3);
    
    const int32_t KH = kernel_h;
    const int32_t KW = kernel_w;
    // view the col output as [N, OH, OW, IC, KH, KW]
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) { // n group c h w
        int _idx = i;
        const int32_t kw = _idx % KW;
        _idx /= KW;
        const int32_t kh = _idx % KH;
        _idx /= KH;
        const int32_t ic = _idx % IC;
        _idx /= IC;
        const int32_t ow = _idx % OW;
        _idx /= OW;
        const int32_t oh = _idx % OH;
        _idx /= OH;
        const int32_t on = _idx;
        
        const int32_t ih = (kh*dilation_h) - pad_t + stride_h*oh;
        const int32_t iw = (kw*dilation_w) - pad_l + stride_w*ow;
        
        if (!is_a_ge_zero_and_a_lt_b(ih, IH) ||
            !is_a_ge_zero_and_a_lt_b(iw, IW) ) { // in padding zone
            *out.unsafe_ptr<float>(i) = 0;
            
        } else {
            const int32_t iidx = on*input.stride_at(0)+ic*input.stride_at(1)+ih*IW+iw;
            *out.unsafe_ptr<float>(i) = input.data_at<float>(iidx);
        } 
    }
}

} // namespace mariana

