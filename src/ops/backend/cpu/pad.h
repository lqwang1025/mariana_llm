/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/pad.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-01:09:06:12
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_PAD_H__
#define __OPS_BACKEND_CPU_PAD_H__

#include <core/tensor.h>
#include <ops/sched_param.h>

namespace mariana {

// (padding_left,  padding_right,
//  padding_top,   padding_bottom,
//  padding_front, padding_back )
void nchw_pad(SchedParam sched_param, const Tensor& input, Tensor& out, uint32_t padding[6], float pad_value);

} // namespace mariana

#endif /* __OPS_BACKEND_CPU_PAD_H__ */

