/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : att_mask.cu
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-18:16:54:46
 * Description:
 * 
 */

#include <ops/att_mask.h>
#include <ops/backend/gpu/impl/att_mask.h>

#include <core/node.h>
#include <core/backend/gpu/cuda_common.h>
#include <core/backend/gpu/cuda_allocator.h>

namespace mariana {

bool AttMaskFunc::plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    cuda_set_device(cuda_ctx->device);
    if (outputs.empty()) {
        outputs.push_back(Tensor(DataOn::GPU));
    }
    outputs[0].try_realloc(inputs[0].dims(), TypeMeta::make<float>());
    return true;
}

bool AttMaskFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    _parallel_sync(m_tp, inputs[0].total_size(), att_mask_cast_to, std::ref(inputs[0]), std::ref(outputs[0]), cuda_ctx);
    return true;
}

} // namespace mariana
