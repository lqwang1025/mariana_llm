/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/layer_norm.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-21:07:43:22
 * Description:
 *
 */

#ifndef __OPS_BACKEND_GPU_IMPL_LAYER_NORM_H__
#define __OPS_BACKEND_GPU_IMPL_LAYER_NORM_H__

#include <core/tensor.h>
#include <ops/sched_param.h>
#include <core/backend/gpu/cuda_common.h>

#define CUDA_LN_BLOCK_SIZE 256

namespace mariana {

struct NormParam;

void group_normlization(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& out, NormParam& norm_param, CUDAContext* cuda_ctx);

void layer_normlization(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, const NormParam& norm_param, Tensor& out, CUDAContext* cuda_ctx);

} // namespace mariana

#endif /* __OPS_BACKEND_GPU_IMPL_LAYER_NORM_H__ */

