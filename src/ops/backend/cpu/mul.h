/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/mul.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-17:10:47:02
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_MUL_H__
#define __OPS_BACKEND_CPU_MUL_H__

#include <core/tensor.h>
#include <ops/sched_param.h>

namespace mariana {

void broadcast_mul2(SchedParam sched_param, const Tensor& a, const Tensor& b, Tensor& out);

void mul_ele(SchedParam sched_param, const Tensor& a, const Tensor& b, Tensor& out);
    
} // namespace mariana

#endif /* __OPS_BACKEND_CPU_MUL_H__ */

