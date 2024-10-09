/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/roll.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-09:10:41:39
 * Description:
 * 
 */

#include <ops/roll.h>
#include <ops/backend/gpu/impl/roll.h>

#include <core/node.h>
#include <core/backend/gpu/cuda_common.h>

namespace mariana {

bool RollFunc::plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    cuda_set_device(cuda_ctx->device);
    if (outputs.empty()) {
        outputs.push_back(Tensor(DataOn::GPU));
    }
    outputs[0].try_realloc(inputs[0].dims(), inputs[0].dtype());
    return true;
}

bool RollFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    _parallel_sync(m_tp, 1, roll4, std::ref(inputs[0]), std::ref(outputs[0]), param, cuda_ctx);
    return true;
}

} // namespace mariana
