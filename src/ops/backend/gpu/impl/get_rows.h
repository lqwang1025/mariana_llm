/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/get_rows.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-14:09:46:26
 * Description:
 *
 */

#ifndef __OPS_BACKEND_GPU_IMPL_GET_ROWS_H__
#define __OPS_BACKEND_GPU_IMPL_GET_ROWS_H__

#include <core/tensor.h>
#include <ops/sched_param.h>
#include <core/backend/gpu/cuda_common.h>

#define CUDA_GET_ROWS_BLOCK_SIZE 256

namespace mariana {
 
void get_rows(SchedParam sched_param, const Tensor& indeices, const Tensor& embedding, Tensor& out, CUDAContext* cuda_ctx);

} // namespace mariana

#endif /* __OPS_BACKEND_GPU_IMPL_GET_ROWS_H__ */

