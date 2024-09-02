/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/mhs_attention.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-25:06:26:21
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_MHS_ATTENTION_H__
#define __OPS_BACKEND_CPU_MHS_ATTENTION_H__

#include <atomic>

#include <core/tensor.h>
#include <ops/sched_param.h>
#include <ops/grounding_dino_encoder_before.h>

namespace mariana {

void mhs_mask_attention_head_split(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& mask, Tensor& out, int32_t n_head, int32_t head_size);

void mhs_mask_attention_batch_split(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& mask, Tensor& out, int32_t n_head, int32_t head_size);

void mhs_attention_batch_split(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& out, int32_t n_head, int32_t head_size);

void mhs_swin_mask_attention_batch_split(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& pos_mask, const Tensor& attn_mask, Tensor& out, int32_t n_head, int32_t head_size);

void bimhs_grdino_qk_dot_batch_split(SchedParam sched_param, const Tensor& Q, const Tensor& K, Tensor& attn_weights, int32_t n_head, int32_t head_size, std::atomic<float>& max_val);

void bimhs_grdino_attn_clamp_bt_split(SchedParam sched_param, Tensor& attn_weights, Tensor& text_attn_weights, const float& max_val, float clamp_min_val, float clamp_max_val);

void bimhs_grdino_attn_v_dot_batch_split(SchedParam sched_param, const Tensor& attn_score, const Tensor& value, Tensor& attn_output, int32_t n_head, int32_t head_size);

void multi_scale_deformable_attention(SchedParam sched_param, const GroundingDinoEncoderBeforeFunc::SpatialShapes& spatial_shapes, const Tensor& value, const Tensor& attention_weights, const Tensor& sampling_offset, const Tensor& reference_points, Tensor& out);

} // namespace marianan

#endif /* __OPS_BACKEND_CPU_MHS_ATTENTION_H__ */

