/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/conv2d.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-26:11:23:02
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_CONV2D_H__
#define __OPS_BACKEND_CPU_CONV2D_H__

#include <core/tensor.h>
#include <ops/sched_param.h>

#include <ops/conv2d.h>

namespace mariana {

void conv2d_element_split(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& out, ConvParam conv_param);

} // namespace mariana

#endif /* __OPS_BACKEND_CPU_CONV2D_H__ */

