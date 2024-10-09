/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/slice.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-10:05:47:33
 * Description:
 *
 */

#ifndef __OPS_BACKEND_GPU_IMPL_SLICE_H__
#define __OPS_BACKEND_GPU_IMPL_SLICE_H__

#include <ops/slice.h>
#include <ops/sched_param.h>

#include <core/tensor.h>
#include <core/backend/gpu/cuda_common.h>

namespace mariana {

#define CUDA_SLICE_BLOCK_SIZE 256

void slice4(SchedParam sched_param, const Tensor& input, Tensor& out, const SliceParam& param, CUDAContext* cuda_ctx);

} // namespace mariana

#endif /* __OPS_BACKEND_GPU_IMPL_SLICE_H__ */

