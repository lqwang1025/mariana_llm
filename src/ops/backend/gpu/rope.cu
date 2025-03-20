/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/roll.cu
 * Authors    : lqwang@pandora
 * Create Time: 2025-03-17:10:41:39
 * Description:
 * 
 */

#include <ops/rope.h>
#include <ops/backend/gpu/impl/rope.h>

#include <core/node.h>
#include <core/backend/gpu/cuda_common.h>

namespace mariana {

bool ROPEFunc::plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    cuda_set_device(cuda_ctx->device);
    if (param.inv_freq.device() == DataOn::CPU) {
        param.inv_freq = param.inv_freq.cuda();
    }
    if (outputs.empty()) {
        Tensor sin(DataOn::GPU);
        Tensor cos(DataOn::GPU);
        outputs = {sin, cos};
    }
    int32_t batch     = inputs[0].dim_at(0);
    int32_t token_len = inputs[0].dim_at(1);
    int32_t head_dim  = 2*param.inv_freq.dim_at(1);
    outputs[0].try_realloc({batch, token_len, head_dim}, TypeMeta::make<float>());
    outputs[1].try_realloc({batch, token_len, head_dim}, TypeMeta::make<float>());
    return true;
}

bool ROPEFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    _parallel_sync(m_tp, 1, rope, std::ref(inputs[0]), std::ref(outputs[0]), std::ref(outputs[1]), param, cuda_ctx);
    return true;
}

} // namespace mariana
