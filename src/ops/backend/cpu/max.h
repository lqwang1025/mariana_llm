/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/max.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-08-20:10:33:30
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_MAX_H__
#define __OPS_BACKEND_CPU_MAX_H__

#include <core/tensor.h>
#include <ops/sched_param.h>

namespace mariana {

void max_last_dim_spilt(SchedParam sched_param, const Tensor& input, Tensor& out);

void topk_index(const int32_t& topk, const Tensor& input, Tensor& out);

} // namespace mariana

#endif /* __OPS_BACKEND_CPU_MAX_H__ */

