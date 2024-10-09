/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/pad.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-09:09:55:45
 * Description:
 *
 */

#ifndef __OPS_BACKEND_GPU_IMPL_PAD_H__
#define __OPS_BACKEND_GPU_IMPL_PAD_H__

#include <core/tensor.h>
#include <ops/sched_param.h>
#include <core/backend/gpu/cuda_common.h>

namespace mariana {

#define CUDA_PAD_BLOCK_SIZE 256

// (padding_left,  padding_right,
//  padding_top,   padding_bottom,
//  padding_front, padding_back )
void nchw_pad(SchedParam sched_param, const Tensor& input, Tensor& out, uint32_t padding[6], float pad_value, CUDAContext* cuda_ctx);

} // namespace mariana

#endif /* __OPS_BACKEND_GPU_IMPL_PAD_H__ */

