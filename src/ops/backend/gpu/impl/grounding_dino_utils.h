/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/grounding_dino_utils.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-04:08:39:38
 * Description:
 *
 */

#ifndef __OPS_BACKEND_GPU_IMPL_GROUNDING_DINO_UTILS_H__
#define __OPS_BACKEND_GPU_IMPL_GROUNDING_DINO_UTILS_H__

#include <vector>
#include <core/tensor.h>
#include <ops/sched_param.h>
#include <ops/grounding_dino_encoder_before.h>
#include <core/backend/gpu/cuda_common.h>

namespace mariana {

#define CUDA_GDENB_BLOCK_SIZE 256
void grounding_dino_encoder_before(SchedParam sched_param, const tensor_list& inputs, const Tensor& level_embed, Tensor& source_flatten, Tensor& lvl_pos_embed_flatten, CUDAContext* cuda_ctx);

} // namespace mariana

#endif /* __OPS_BACKEND_GPU_IMPL_GROUNDING_DINO_UTILS_H__ */

