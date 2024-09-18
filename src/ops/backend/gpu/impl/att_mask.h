/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/att_mask.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-18:17:10:20
 * Description:
 *
 */

#ifndef __OPS_BACKEND_GPU_IMPL_ATT_MASK_H__
#define __OPS_BACKEND_GPU_IMPL_ATT_MASK_H__

#include <core/tensor.h>
#include <ops/sched_param.h>
#include <core/backend/gpu/cuda_common.h>

#define CUDA_ATT_MASK_CAST_BLOCK_SIZE 256

namespace mariana {
 
void att_mask_cast_to(SchedParam sched_param, const Tensor& input, Tensor& out, CUDAContext* cuda_ctx);

} // namespace mariana

#endif /* __OPS_BACKEND_GPU_IMPL_ATT_MASK_H__ */

