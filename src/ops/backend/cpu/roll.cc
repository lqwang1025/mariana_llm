/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/roll.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-03:09:18:15
 * Description:
 * 
 */

#include <ops/backend/cpu/roll.h>

namespace mariana {

void roll4(SchedParam sched_param, const Tensor& input, Tensor& out, const RollParam& param) {
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        uint32_t idx = i;
        uint32_t idx4 = idx % input.dim_at(3);
        idx /= input.dim_at(3);
        uint32_t idx3 = idx % input.dim_at(2);
        idx /= input.dim_at(2);
        uint32_t idx2 = idx % input.dim_at(1);
        idx /= input.dim_at(1);
        uint32_t idx1 = idx;
        int32_t idxes[4] = {(int32_t)idx1, (int32_t)idx2, (int32_t)idx3, (int32_t)idx4};

        for (size_t i = 0; i < param.dims.size(); ++i) {
            idxes[param.dims[i]] += param.shifts[i];
            if (out.dim_at(param.dims[i]) <= idxes[param.dims[i]]) {
                idxes[param.dims[i]] -= out.dim_at(param.dims[i]);
            } else if (idxes[param.dims[i]] < 0) {
                idxes[param.dims[i]] += out.dim_at(param.dims[i]);
            } else {
                // do nothing
            }
        }
        
        uint32_t out_idx =
            idxes[0]*input.stride_at(0)+idxes[1]*input.stride_at(1)+
            idxes[2]*input.stride_at(2)+idxes[3]*input.stride_at(3);
        *out.unsafe_ptr<float>(out_idx) = input.data_at<float>(i);
    }
}

} // namespace mariana
