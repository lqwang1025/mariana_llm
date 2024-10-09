/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/permute.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-09:13:24:56
 * Description:
 *
 */

#ifndef __OPS_BACKEND_GPU_IMPL_PERMUTE_H__
#define __OPS_BACKEND_GPU_IMPL_PERMUTE_H__

#include <cstdint>

#include <ops/sched_param.h>

#include <core/tensor.h>
#include <core/backend/gpu/cuda_common.h>

namespace mariana {

#define CUDA_PERMUTE_BLOCK_SIZE 256

void permute4(SchedParam sched_param, const Tensor& input, Tensor& out, uint8_t perms[4], CUDAContext* cuda_ctx);

void permute6(SchedParam sched_param, const Tensor& input, Tensor& out, uint8_t perms[6], CUDAContext* cuda_ctx);

} // namespace mariana

#endif /* __OPS_BACKEND_GPU_IMPL_PERMUTE_H__ */

