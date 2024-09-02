/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : conv2d.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-26:11:24:46
 * Description:
 * 
 */

#include <ops/backend/cpu/conv2d.h>

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

void conv2d_element_split(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& out, ConvParam conv_param) {
    const int32_t OC = out.dim_at(1)/conv_param.groups;
    const int32_t OH = out.dim_at(2);
    const int32_t OW = out.dim_at(3);
    
    const int32_t IC = input.dim_at(1)/conv_param.groups;
    const int32_t IH = input.dim_at(2);
    const int32_t IW = input.dim_at(3);
    
    const int32_t KH = weight.dim_at(2);
    const int32_t KW = weight.dim_at(3);
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) { // n group c h w
        int _idx = i;
        const int32_t w = _idx % OW;
        _idx /= OW;
        const int32_t h = _idx % OH;
        _idx /= OH;
        const int32_t c = _idx % OC;
        _idx /= OC;
        const int32_t g = _idx % conv_param.groups;
        _idx /= conv_param.groups;
        const int32_t b = _idx;

        const int32_t ih_start = (h*conv_param.strides[1]) - conv_param.padding[1];
        const int32_t iw_start = (w*conv_param.strides[0]) - conv_param.padding[0];
        
        float sum = 0.f;
        for (int32_t ic = 0; ic < IC; ++ic) { // kernel begin
            for (int32_t kh = 0; kh < KH; ++kh) {
                const int32_t cur_y = ih_start + conv_param.dilation[1]*kh;
                if (is_a_ge_zero_and_a_lt_b(cur_y, IH)) { // (cur_y >= 0) && (cur_y < IH)
                    for (int32_t kw = 0; kw < KW; ++kw) {
                        const int32_t cur_x = iw_start + conv_param.dilation[0]*kw;
                        if (is_a_ge_zero_and_a_lt_b(cur_x, IW)) { // (cur_x >= 0) && (cur_x < IW)
                            const int32_t iidx = b*conv_param.groups*input.stride_at(0)+g*input.stride_at(0)+ic*input.stride_at(1)+cur_y*input.stride_at(2)+cur_x;
                            const int32_t widx = g*OC*weight.stride_at(0)+c*weight.stride_at(0)+ic*weight.stride_at(1)+kh*KW+kw;
                            sum += input.data_at<float>(iidx)*weight.data_at<float>(widx);
                        }
                    }
                }
            }
        }  // kernel end
        *out.unsafe_ptr<float>(i) = sum+bias.data_at<float>(c);
    }
}

} // namespace mariana
 
