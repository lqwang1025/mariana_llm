/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/swin_patch_merging.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-07-05:06:33:28
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_SWIN_PATCH_MERGING_H__
#define __OPS_BACKEND_CPU_SWIN_PATCH_MERGING_H__

#include <core/tensor.h>
#include <ops/sched_param.h>

namespace mariana {

void swin_patch_merge(SchedParam sched_param, const Tensor& input, Tensor& out, int32_t step);

} // namespace mariana

#endif /* __OPS_BACKEND_CPU_SWIN_PATCH_MERGING_H__ */

