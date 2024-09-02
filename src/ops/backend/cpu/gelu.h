/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/gelu.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-25:16:54:16
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_GELU_H__
#define __OPS_BACKEND_CPU_GELU_H__

#include <core/tensor.h>
#include <ops/sched_param.h>

namespace mariana {

void gelu(SchedParam sched_param, const Tensor& input, Tensor& out);

float gelu_single(float x);

float relu_single(float x);

} // namespace mariana

#endif /* __OPS_BACKEND_CPU_GELU_H__ */

