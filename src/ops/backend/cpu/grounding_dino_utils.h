/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/grounding_dino_utils.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-10:10:08:36
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_GROUNDING_DINO_UTILS_H__
#define __OPS_BACKEND_CPU_GROUNDING_DINO_UTILS_H__

#include <vector>
#include <core/tensor.h>
#include <ops/sched_param.h>

namespace mariana {

using tensor_list = std::vector<Tensor>;
struct ExeContext;

void grounding_dino_encoder_before(SchedParam sched_param, const tensor_list& inputs, const Tensor& level_embed, Tensor& source_flatten, Tensor& lvl_pos_embed_flatten);

void generate_encoder_output_proposals(SchedParam sched_param, ExeContext& context, const Tensor& enc_output, Tensor& output_proposals, Tensor& object_query);

void decoder_reference_points_correct(SchedParam sched_param, const Tensor& tmp, const Tensor& reference_points, Tensor& out, float eps);

void bbox_center_to_corners(SchedParam sched_param, const Tensor& src, int32_t img_h, int32_t img_w, Tensor& out);

void grounding_dino_pre_process(SchedParam sched_param, const Tensor& src, const std::vector<float>& means, const std::vector<float>& stds, Tensor& out);

} // namespace mariana

#endif /* __OPS_BACKEND_CPU_GROUNDING_DINO_UTILS_H__ */

