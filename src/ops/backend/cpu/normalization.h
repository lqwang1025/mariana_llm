/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/normalization.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-22:01:00:29
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_NORMALIZATION_H__
#define __OPS_BACKEND_CPU_NORMALIZATION_H__

#include <core/tensor.h>
#include <ops/sched_param.h>

namespace mariana {

struct NormParam;

void group_normlization(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& out, NormParam& norm_param);

void layer_normlization(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& out, NormParam& norm_param);

} // namespace mariana

#endif /* __OPS_BACKEND_CPU_NORMALIZATION_H__ */

