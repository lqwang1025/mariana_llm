/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/tile.h
 * Authors    : lqwang@SMT23090002
 * Create Time: 2025-03-24:20:16:31
 * Description:
 *
 */

#ifndef __OPS_BACKEND_GPU_IMPL_TILE_H__
#define __OPS_BACKEND_GPU_IMPL_TILE_H__

#include <ops/sched_param.h>
#include <core/tensor.h>
#include <core/backend/gpu/cuda_common.h>

namespace mariana {

#define CUDA_TILE_BLOCK_SIZE 256

void tile4(SchedParam sched_param, const Tensor& input, Tensor& out, uint32_t repeats[4], CUDAContext* cuda_ctx);

} // namespace mariana


#endif /* __OPS_BACKEND_GPU_IMPL_TILE_H__ */

