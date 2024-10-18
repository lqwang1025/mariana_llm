/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/grounding_dino_sine_position_embedding.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-15:09:57:58
 * Description:
 *
 */

#ifndef __OPS_BACKEND_GPU_IMPL_GROUNDING_DINO_SINE_POSITION_EMBEDDING_H__
#define __OPS_BACKEND_GPU_IMPL_GROUNDING_DINO_SINE_POSITION_EMBEDDING_H__

#include <core/tensor.h>
#include <ops/sched_param.h>
#include <core/backend/gpu/cuda_common.h>

namespace mariana {

#define CUDA_GDSPE_BLOCK_SIZE 256

void grounding_dino_sine_position_embedding(SchedParam sched_param, const Tensor& input, Tensor& out, float scale, float temperature, CUDAContext* cuda_ctx);

} // namespace mariana

#endif /* __OPS_BACKEND_GPU_IMPL_GROUNDING_DINO_SINE_POSITION_EMBEDDING_H__ */

