/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/rms_norm.cu
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-21:06:17:21
 * Description:
 * 
 */

#include <core/node.h>
#include <ops/rms_norm.h>
#include <core/backend/gpu/cuda_common.h>
#include <ops/backend/gpu/impl/layer_norm.h>

namespace mariana {

bool RMSNormFunc::plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (inputs.size() != 1) {
        MLOG(ERROR)<<"RMSNorm input's size must be 1 now:"<<inputs.size();
        return false;
    }
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    cuda_set_device(cuda_ctx->device);
    if (m_weight.device() != DataOn::GPU) {
        m_weight = m_weight.cuda(cuda_ctx->stream());
    }
    if (outputs.empty()) {
        outputs.push_back(Tensor(DataOn::GPU));
    }
    outputs[0].try_realloc(inputs[0].dims(), inputs[0].dtype());
    return true;
}

bool RMSNormFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    _parallel_sync(m_tp, inputs[0].dim_at(0), RMS_normlization, std::ref(inputs[0]),
                   std::ref(m_weight), m_epsilon, std::ref(outputs[0]), cuda_ctx);
    return true;
}

} // namespace mariana
