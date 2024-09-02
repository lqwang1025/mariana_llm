/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/im2col.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-26:15:31:34
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_IM2COL_H__
#define __OPS_BACKEND_CPU_IM2COL_H__

#include <cstdint>

#include <core/tensor.h>
#include <ops/sched_param.h>

namespace mariana {

void im2col_element_split(SchedParam sched_param, const Tensor& input, Tensor& out, int32_t kernel_h, int32_t kernel_w, int32_t pad_t, int32_t pad_l, int32_t pad_b, int32_t pad_r, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, int32_t groups);

} // namespace mariana

#endif /* __OPS_BACKEND_CPU_IM2COL_H__ */

