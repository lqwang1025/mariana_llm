/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/mhs_attention.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-09-27:10:29:32
 * Description:
 *
 */

#ifndef __OPS_BACKEND_GPU_IMPL_MHS_ATTENTION_H__
#define __OPS_BACKEND_GPU_IMPL_MHS_ATTENTION_H__

#include <core/tensor.h>
#include <ops/sched_param.h>
#include <core/backend/gpu/cuda_common.h>

#define CUDA_ATTN_BLOCK_SIZE 256

namespace mariana {

void mhs_mask_attention(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& mask, Tensor& out, int32_t n_head, int32_t head_size, CUDAContext* cuda_ctx);

void mhs_attention(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& out, int32_t n_head, int32_t head_size, CUDAContext* cuda_ctx);

void mhs_swin_mask_attention(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& pos_mask, const Tensor& attn_mask, Tensor& out, int32_t n_head, int32_t head_size, CUDAContext* cuda_ctx);

} // namespace mariana

#endif /* __OPS_BACKEND_GPU_IMPL_MHS_ATTENTION_H__ */

