/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/permute.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-24:09:09:43
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_PERMUTE_H__
#define __OPS_BACKEND_CPU_PERMUTE_H__

#include <cstdint>

#include <core/tensor.h>
#include <ops/sched_param.h>

namespace mariana {

void permute4(SchedParam sched_param, const Tensor& input, Tensor& out, uint8_t perms[4]);

void permute6(SchedParam sched_param, const Tensor& input, Tensor& out, uint8_t perms[6]);

} // namespace mariana

#endif /* __OPS_BACKEND_CPU_PERMUTE_H__ */

