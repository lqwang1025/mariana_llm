/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/softmax.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-31:08:58:18
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_SOFTMAX_H__
#define __OPS_BACKEND_CPU_SOFTMAX_H__

#include <cstdint>

#include <core/tensor.h>
#include <ops/sched_param.h>

namespace mariana {

void softmax(SchedParam sched_param, const Tensor& input, Tensor& out);

} // namespace mariana

#endif /* __OPS_BACKEND_CPU_SOFTMAX_H__ */

