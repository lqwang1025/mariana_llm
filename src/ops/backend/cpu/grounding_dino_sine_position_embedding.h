/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/grounding_dino_sine_position_embedding.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-08:11:03:04
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_GROUNDING_DINO_SINE_POSITION_EMBEDDING_H__
#define __OPS_BACKEND_CPU_GROUNDING_DINO_SINE_POSITION_EMBEDDING_H__

#include <core/tensor.h>
#include <ops/sched_param.h>

namespace mariana {

void grounding_dino_sine_position_embedding(SchedParam sched_param, const Tensor& input, Tensor& out, float scale, float temperature);

void grounding_dino_get_text_enhancer_sine_pos_embed(SchedParam sched_param, const Tensor& input, Tensor& out, float scale, float temperature,  bool exchange_xy);

} // namespace mariana

#endif /* __OPS_BACKEND_CPU_GROUNDING_DINO_SINE_POSITION_EMBEDDING_H__ */

