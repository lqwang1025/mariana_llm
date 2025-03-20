/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/rope.h
 * Authors    : lqwang@SMT23090002
 * Create Time: 2025-03-20:13:55:01
 * Description:
 *
 */

#ifndef __OPS_BACKEND_GPU_IMPL_ROPE_H__
#define __OPS_BACKEND_GPU_IMPL_ROPE_H__

#include <ops/rope.h>
#include <ops/sched_param.h>

#include <core/tensor.h>
#include <core/backend/gpu/cuda_common.h>

namespace mariana {

#define CUDA_ROPE_BLOCK_SIZE 256

void rope(SchedParam sched_param, const Tensor& input, Tensor& sin, Tensor& cos, const ROPEParam& param, CUDAContext* cuda_ctx);

} // namespace mariana

#endif /* __OPS_BACKEND_GPU_IMPL_ROPE_H__ */

