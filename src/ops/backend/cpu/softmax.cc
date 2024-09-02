/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/softmax.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-31:08:59:29
 * Description:
 * 
 */

#include <cmath>
#include <cfloat>

#include <ops/backend/cpu/softmax.h>

namespace mariana {

void softmax(SchedParam sched_param, const Tensor& input, Tensor& out) {
    uint32_t ele_size = static_cast<uint32_t>(out.dim_at(out.dim_size()-1));
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        uint32_t offset = i*ele_size;
        float expsum = 0.f;
        for (uint32_t n = 0; n < ele_size; ++n) {
            float item = input.data_at<float>(offset+n);
            float expv = expf(item);
            expsum += expv;
            *out.unsafe_ptr<float>(offset+n) = expv;
        }
        float expsum_inv = expsum == 0.f ? 0.f : 1.f/expsum;
        for (uint32_t n = 0; n < ele_size; ++n) {
            *out.unsafe_ptr<float>(offset+n) *= expsum_inv;
        }
    }
}

} // namespace mariana
