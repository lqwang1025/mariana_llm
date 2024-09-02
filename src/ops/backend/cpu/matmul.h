/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/matmul.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-06-23:19:23:39
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_MATMUL_H__
#define __OPS_BACKEND_CPU_MATMUL_H__

#include <core/tensor.h>
#include <ops/ops.h>
#include <ops/sched_param.h>

namespace mariana {

// thread split by rows

// c = alpha*A*B+ beta*bias
void matmul(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& out, float alpha, float beta, OpCategory act_cate);

} // namespace mariana

#endif /* __OPS_BACKEND_CPU_MATMUL_H__ */

