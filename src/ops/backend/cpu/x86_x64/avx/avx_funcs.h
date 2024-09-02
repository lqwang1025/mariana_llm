/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/x86_x64/avx/avx_funcs.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-06-23:22:20:03
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_X86_X64_AVX_AVX_FUNCS_H__
#define __OPS_BACKEND_CPU_X86_X64_AVX_AVX_FUNCS_H__

#include <cstdint>

#include <ops/ops.h>
#include <ops/sched_param.h>
#include <ops/layer_norm.h>

namespace mariana {

class Tensor;

void _avx2_gemm_fp32(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& out, float alpha, float beta, OpCategory act_cate);

void _avx2_gemm_no_bias_fp32(SchedParam sched_param, const Tensor& input, const Tensor& weight, Tensor& out, float alpha, OpCategory act_cate);

void _avx2_mhs_mask_attention_batch_split_fp32(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& mask, Tensor& out, int32_t n_head, int32_t head_size);

void _avx2_mhs_attention_batch_split_fp32(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& out, int32_t n_head, int32_t head_size);

void _avx2_mhs_swin_mask_attention_batch_split_fp32(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& mask, const Tensor& attn_mask, Tensor& out, int32_t n_head, int32_t head_size);

void _avx2_bimhs_grdino_qk_dot_batch_split(SchedParam sched_param, const Tensor& Q, const Tensor& K, Tensor& attn_weights, int32_t n_head, int32_t head_size, std::atomic<float>& max_val);

void _avx2_bimhs_grdino_attn_clamp_bt_split(SchedParam sched_param, Tensor& attn_weights, Tensor& text_attn_weights, const float& max_val, float clamp_min_val, float clamp_max_val);

void _avx2_bimhs_grdino_attn_v_dot_batch_split(SchedParam sched_param, const Tensor& attn_score, const Tensor& value, Tensor& attn_output, int32_t n_head, int32_t head_size);

void _avx_GELU_fp32(SchedParam sched_param, const Tensor& input, Tensor& out);

void _avx_norm_fp32(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& out, NormParam& norm_param);

} // namespace mariana

#endif /* __OPS_BACKEND_CPU_X86_X64_AVX_AVX_FUNCS_H__ */

