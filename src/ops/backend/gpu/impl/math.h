/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/math.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-19:05:32:39
 * Description:
 *
 */

#ifndef __OPS_BACKEND_GPU_IMPL_MATH_H__
#define __OPS_BACKEND_GPU_IMPL_MATH_H__

#include <core/tensor.h>
#include <ops/sched_param.h>
#include <core/backend/gpu/cuda_common.h>

#define CUDA_ADD_BLOCK_SIZE 256

namespace mariana {
 
void add_ele(SchedParam sched_param, const Tensor& a, const Tensor& b, Tensor& out, CUDAContext* cuda_ctx);

} // namespace mariana

#endif /* __OPS_BACKEND_GPU_IMPL_MATH_H__ */

