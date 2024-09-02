/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/add.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-21:19:37:21
 * Description:
 * 
 */

#include <ops/backend/cpu/add.h>
#include <utils/mariana_define.h>

namespace mariana {

void add_ele(SchedParam sched_param, const Tensor& a, const Tensor& b, Tensor& out) {
    TRACE();
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        float a_item = a.data_at<float>(i);
        float b_item = b.data_at<float>(i);
        *out.unsafe_ptr<float>(i) = a_item + b_item;
    }
}
    
} // namespace mariana
