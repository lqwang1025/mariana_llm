/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/matmul.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-09-26:17:27:17
 * Description:
 *
 */

#ifndef __OPS_BACKEND_GPU_IMPL_MATMUL_H__
#define __OPS_BACKEND_GPU_IMPL_MATMUL_H__

#include <ops/ops.h>
#include <core/tensor.h>
#include <ops/sched_param.h>
#include <core/backend/gpu/cuda_common.h>

#define CUDA_MATMUL_BLOCK_SIZE 256

namespace mariana {

// c = alpha*A*B+ beta*bias
void matmul(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& out, float alpha, float beta, OpCategory act_cate, CUDAContext* cuda_ctx);

} // namespace mariana

#endif /* __OPS_BACKEND_GPU_IMPL_MATMUL_H__ */

