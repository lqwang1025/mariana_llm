/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/slice.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-02:16:58:38
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_SLICE_H__
#define __OPS_BACKEND_CPU_SLICE_H__

#include <cstdint>

#include <ops/slice.h>
#include <core/tensor.h>
#include <ops/sched_param.h>

namespace mariana {

void slice4(SchedParam sched_param, const Tensor& input, Tensor& out, const SliceParam& param);

} // namespace mariana

#endif /* __OPS_BACKEND_CPU_SLICE_H__ */

