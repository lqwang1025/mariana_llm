/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/im2col.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-08:08:19:45
 * Description:
 *
 */

#ifndef __OPS_BACKEND_GPU_IMPL_IM2COL_H__
#define __OPS_BACKEND_GPU_IMPL_IM2COL_H__

#include <core/tensor.h>
#include <ops/sched_param.h>
#include <core/backend/gpu/cuda_common.h>

#define CUDA_IMG2COL_BLOCK_SIZE 256

namespace mariana {

void im2col(SchedParam sched_param, const Tensor& input, Tensor& out, int32_t kernel_h, int32_t kernel_w, int32_t pad_t, int32_t pad_l, int32_t pad_b, int32_t pad_r, int32_t stride_h, int32_t stride_w, int32_t dilation_h, int32_t dilation_w, int32_t groups, CUDAContext* cuda_ctx);

} // namespace mariana

#endif /* __OPS_BACKEND_GPU_IMPL_IM2COL_H__ */

