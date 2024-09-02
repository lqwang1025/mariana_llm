/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : mul.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-17:13:08:18
 * Description:
 * 
 */

#include <ops/backend/cpu/mul.h>

namespace mariana {

void broadcast_mul2(SchedParam sched_param, const Tensor& a, const Tensor& b, Tensor& out) {
    TRACE();
    const int32_t dim1 = a.dim_at(0);
    const int32_t dim2 = a.dim_at(1);
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        uint32_t idx1 = i / dim2;
        uint32_t idx2 = i % dim2;
        float a_item = a.data_at<float>(i);
        float b_item = 1.f;
        if (b.dim_at(0) == dim1) {
            b_item = b.data_at<float>(idx1);
        } else {
            b_item = b.data_at<float>(idx2);
        }
        *out.unsafe_ptr<float>(i) = a_item * b_item;
    }
}

void mul_ele(SchedParam sched_param, const Tensor& a, const Tensor& b, Tensor& out) {
    TRACE();
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        float a_item = a.data_at<float>(i);
        float b_item = b.data_at<float>(i);
        *out.unsafe_ptr<float>(i) = a_item * b_item;
    }
}

} // namespace mariana
