/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/sigmoid.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-08-21:11:08:26
 * Description:
 * 
 */

#include <cmath>

#include <ops/backend/cpu/sigmoid.h>

namespace mariana {

void sigmoid(SchedParam sched_param, const Tensor& input, Tensor& out) {
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        float item = input.data_at<float>(i);
        *out.unsafe_ptr<float>(i) = 1.f/(1+expf(-item));
    }
}

} // namespace mariana
