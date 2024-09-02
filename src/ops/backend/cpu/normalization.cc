/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : normalization.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-22:01:00:34
 * Description:
 * 
 */

#include <cmath>

#include <ops/layer_norm.h>
#include <ops/backend/cpu/normalization.h>
#include <ops/backend/cpu/x86_x64/avx/avx_funcs.h>

#include <utils/mariana_define.h>

namespace mariana {

[[maybe_unused]]static void _naive_layer_normalization(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& out, NormParam& norm_param) {
    const int32_t nbs = input.stride_at(1);
    
    for (uint32_t row = sched_param.this_thread_begin_index(); row < sched_param.this_thread_end_index(); ++row) {
        float mean = 0.f;
        
        for (int32_t col = 0; col < nbs; ++col) {
            mean += input.data_at<float>(row*nbs+col);
        }
        mean /= nbs;
        
        float var  = 0.f;
        for (int32_t col = 0; col < nbs; ++col) {
            float x_shift = input.data_at<float>(row*nbs+col)- mean;
            var += std::pow(x_shift, 2);
        }
        var /= nbs;
        var = 1.f / sqrtf(var + norm_param.epsilon);
        for (int32_t col = 0; col < nbs; ++col) {
            float val = input.data_at<float>(row*nbs+col);
            float n = (var*(val - mean));
            float o = n*weight.data_at<float>(col) + bias.data_at<float>(col);
            *out.unsafe_ptr<float>(row*nbs+col) = o;
        }
    }
}


[[maybe_unused]]static void _naive_group_normalization(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& out, NormParam& norm_param) {
    const int32_t G   = input.dim_at(1);
    const int32_t nbs = input.stride_at(1);
    const int32_t GC  = input.dim_at(2);
    const int32_t HW  = input.dim_at(3);
    
    for (uint32_t row = sched_param.this_thread_begin_index(); row < sched_param.this_thread_end_index(); ++row) {
        float mean = 0.f;
        
        for (int32_t col = 0; col < nbs; ++col) {
            mean += input.data_at<float>(row*nbs+col);
        }
        mean /= nbs;
        
        float var  = 0.f;
        for (int32_t col = 0; col < nbs; ++col) {
            float x_shift = input.data_at<float>(row*nbs+col)- mean;
            var += std::pow(x_shift, 2);
        }
        var /= nbs;
        var = 1.f / sqrtf(var + norm_param.epsilon);
        
        const int32_t g = row%G;
        for (int32_t col = 0; col < nbs; ++col) {
            const int32_t gc = (col/HW) % GC;
            float val = input.data_at<float>(row*nbs+col);
            float n = (var*(val - mean));
            float o = n*weight.data_at<float>(g*GC+gc) + bias.data_at<float>(g*GC+gc);
            *out.unsafe_ptr<float>(row*nbs+col) = o;
        }
    }
}

void layer_normlization(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& out, NormParam& norm_param) {
    _naive_layer_normalization(sched_param, input, weight, bias, out, norm_param);
    //_avx_norm_fp32(sched_param, input, weight, bias, out, norm_param);
}

void group_normlization(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& out, NormParam& norm_param) {
    _naive_group_normalization(sched_param, input, weight, bias, out, norm_param);
}

} // namespace mariana
