/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/gelu.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-25:16:59:38
 * Description:
 * 
 */

#include <cmath>

#include <ops/backend/cpu/gelu.h>
#include <utils/mariana_define.h>
#include <ops/backend/cpu/x86_x64/avx/avx_funcs.h>

#define GELU_SCALING_FACTOR sqrtf(0.5f)

namespace mariana {

static void _naive_gelu_fp32(SchedParam sched_param, const Tensor& input, Tensor& out) {
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        float x = input.data_at<float>(i);
        *out.unsafe_ptr<float>(i) = 0.5f*x*(1.f+erff(GELU_SCALING_FACTOR*x));
    }
}

float gelu_single(float x) {
    return 0.5f*x*(1.f+erff(GELU_SCALING_FACTOR*x));
}

float relu_single(float x) {
    return MAX(x, 0.f);
}

void gelu(SchedParam sched_param, const Tensor& input, Tensor& out) {
    _naive_gelu_fp32(sched_param, input, out);
    // _avx_GELU_fp32(sched_param, input, out);
}

} // namespace mariana
