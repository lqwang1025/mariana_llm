/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/permute.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-24:09:27:56
 * Description:
 * 
 */

#include <ops/backend/cpu/permute.h>

#include <utils/mariana_define.h>

namespace mariana {

void permute4(SchedParam sched_param, const Tensor& input, Tensor& out, uint8_t perms[4]) {
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        uint32_t idx = i;
        uint32_t idx4 = idx % input.dim_at(3);
        idx /= input.dim_at(3);
        uint32_t idx3 = idx % input.dim_at(2);
        idx /= input.dim_at(2);
        uint32_t idx2 = idx % input.dim_at(1);
        idx /= input.dim_at(1);
        uint32_t idx1 = idx;
        const uint32_t idxes[4] = {idx1, idx2, idx3, idx4};
        uint32_t dst_idx =
            idxes[perms[0]]*out.stride_at(0)+idxes[perms[1]]*out.stride_at(1)+
            idxes[perms[2]]*out.stride_at(2)+idxes[perms[3]]*out.stride_at(3);
        *out.unsafe_ptr<float>(dst_idx) = input.data_at<float>(i);
    }
}

void permute6(SchedParam sched_param, const Tensor& input, Tensor& out, uint8_t perms[6]) {
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        uint32_t idx = i;
        uint32_t idx6 = idx % input.dim_at(5);
        idx /= input.dim_at(5);
        uint32_t idx5 = idx % input.dim_at(4);
        idx /= input.dim_at(4);
        uint32_t idx4 = idx % input.dim_at(3);
        idx /= input.dim_at(3);
        uint32_t idx3 = idx % input.dim_at(2);
        idx /= input.dim_at(2);
        uint32_t idx2 = idx % input.dim_at(1);
        idx /= input.dim_at(1);
        uint32_t idx1 = idx;
        const uint32_t idxes[6] = {idx1, idx2, idx3, idx4, idx5, idx6};
        uint32_t dst_idx =
            idxes[perms[0]]*out.stride_at(0)+idxes[perms[1]]*out.stride_at(1)+
            idxes[perms[2]]*out.stride_at(2)+idxes[perms[3]]*out.stride_at(3)+
            idxes[perms[4]]*out.stride_at(4)+idxes[perms[5]]*out.stride_at(5);
        *out.unsafe_ptr<float>(dst_idx) = input.data_at<float>(i);
    }
}

} // namespace mariana
