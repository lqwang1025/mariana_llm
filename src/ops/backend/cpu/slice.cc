/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/slice.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-02:17:00:41
 * Description:
 * 
 */

#include <ops/backend/cpu/slice.h>

#include <utils/mariana_define.h>

namespace mariana {

void slice4(SchedParam sched_param, const Tensor& input, Tensor& out, const SliceParam& param) {
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        uint32_t idx = i;
        uint32_t idx4 = idx % out.dim_at(3);
        idx /= out.dim_at(3);
        uint32_t idx3 = idx % out.dim_at(2);
        idx /= out.dim_at(2);
        uint32_t idx2 = idx % out.dim_at(1);
        idx /= out.dim_at(1);
        uint32_t idx1 = idx;
        uint32_t idxes[4]  = {idx1, idx2, idx3, idx4};
        for (size_t i = 0; i < param.axes.size(); ++i) {
            idxes[param.axes[i]] *= param.steps[i];
            idxes[param.axes[i]] += param.starts[i];
        }
        uint32_t input_idx =
            idxes[0]*input.stride_at(0)+idxes[1]*input.stride_at(1)+
            idxes[2]*input.stride_at(2)+idxes[3]*input.stride_at(3);
        *out.unsafe_ptr<float>(i) = input.data_at<float>(input_idx);
    }
}

} // namespace mariana
