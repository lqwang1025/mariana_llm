/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/swin_stage_output.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-11:14:20:32
 * Description:
 * 
 */

#include <ops/layer_norm.h>
#include <ops/swin_stage_output.h>
#include <ops/backend/gpu/impl/permute.h>

#include <core/node.h>
#include <core/backend/gpu/cuda_common.h>

namespace mariana {

bool SwinStageOutputFunc::plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    cuda_set_device(cuda_ctx->device);
    if (m_ln_out.total_size() == 0) {
        m_ln_out = Tensor(DataOn::GPU);
    }
    tensor_list __outputs = {m_ln_out};
    m_layer_norm->plan_forward_gpu(inputs, __outputs, context);
    if (outputs.empty()) {
        outputs.push_back(DataOn::GPU);
    }
    outputs[0].try_realloc({m_ln_out.dim_at(0), m_ln_out.dim_at(2),
            (int32_t)m_owner->info_shared_nodes()[0]->runtime_info().feature_height,
            (int32_t)m_owner->info_shared_nodes()[0]->runtime_info().feature_width}, inputs[0].dtype());
    return true;
}

bool SwinStageOutputFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    tensor_list __outputs = {m_ln_out};
    m_layer_norm->_forward_gpu(inputs, __outputs, context);
    uint8_t perms[4] = {0, 3, 1, 2};
    m_ln_out.reshape({m_ln_out.dim_at(0), (int32_t)m_owner->info_shared_nodes()[0]->runtime_info().feature_height,
            (int32_t)m_owner->info_shared_nodes()[0]->runtime_info().feature_width, m_ln_out.dim_at(2)});
    _parallel_sync(m_tp, 1, permute4, std::ref(m_ln_out), std::ref(outputs[0]), perms, cuda_ctx);
    return true;
}

} // namespace mariana
