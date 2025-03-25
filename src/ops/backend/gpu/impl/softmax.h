/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/softmax.h
 * Authors    : lqwang@SMT23090002
 * Create Time: 2025-03-25:14:59:10
 * Description:
 *
 */

#ifndef __OPS_BACKEND_GPU_IMPL_SOFTMAX_H__
#define __OPS_BACKEND_GPU_IMPL_SOFTMAX_H__

#include <ops/sched_param.h>
#include <core/tensor.h>
#include <core/backend/gpu/cuda_common.h>

namespace mariana {

#define CUDA_SOFTMAX_BLOCK_SIZE 256

void softmax3(SchedParam sched_param, const Tensor& input, Tensor& out, int32_t dim, CUDAContext* cuda_ctx);

} // namespace mariana

#endif /* __OPS_BACKEND_GPU_IMPL_SOFTMAX_H__ */

