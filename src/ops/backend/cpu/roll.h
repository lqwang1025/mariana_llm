/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/roll.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-03:09:16:52
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_ROLL_H__
#define __OPS_BACKEND_CPU_ROLL_H__

#include <cstdint>

#include <ops/roll.h>
#include <core/tensor.h>
#include <ops/sched_param.h>

namespace mariana {

void roll4(SchedParam sched_param, const Tensor& input, Tensor& out, const RollParam& param);

} // namespace mariana

#endif /* __OPS_BACKEND_CPU_ROLL_H__ */

