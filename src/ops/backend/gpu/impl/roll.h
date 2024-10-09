/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/roll.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-09:10:44:50
 * Description:
 *
 */

#ifndef __OPS_BACKEND_GPU_IMPL_ROLL_H__
#define __OPS_BACKEND_GPU_IMPL_ROLL_H__

#include <ops/roll.h>
#include <ops/sched_param.h>

#include <core/tensor.h>
#include <core/backend/gpu/cuda_common.h>

namespace mariana {

#define CUDA_ROLL_BLOCK_SIZE 256

void roll4(SchedParam sched_param, const Tensor& input, Tensor& out, const RollParam& param, CUDAContext* cuda_ctx);

} // namespace mariana

#endif /* __OPS_BACKEND_GPU_IMPL_ROLL_H__ */

