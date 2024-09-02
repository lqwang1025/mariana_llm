/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/max.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-08-20:10:35:04
 * Description:
 * 
 */

#include <numeric>
#include <vector>
#include <cfloat>
#include <algorithm>

#include <core/impl/allocator.h>

#include <utils/mariana_define.h>

#include <ops/backend/cpu/max.h>

namespace mariana {

void max_last_dim_spilt(SchedParam sched_param, const Tensor& input, Tensor& out) {
    uint32_t C = (uint32_t)input.dim_at(2);
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        float max = -FLT_MAX;
        uint32_t offset = i*input.stride_at(1);
        for (uint32_t c = 0;  c < C; ++c) {
            float __tmp = input.data_at<float>(offset+c);
            if (__tmp > max) {
                max = __tmp;
            }
        }
        *out.unsafe_ptr<float>(i) = max;
    }
}

void topk_index(const int32_t& topk, const Tensor& input, Tensor& out) {
    std::vector<int32_t> idx(input.total_size()-1);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&input](int32_t a, int32_t b) {
        return input.data_at<float>(a) > input.data_at<float>(b);
    });
    for (int32_t i = 0; i < topk; ++i) {
        *out.unsafe_ptr<int32_t>(i) = idx.at(i);
    }
}

} // namespace mariana
