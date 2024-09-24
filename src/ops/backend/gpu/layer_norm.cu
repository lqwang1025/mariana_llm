/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/layer_norm.cu
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-21:06:17:21
 * Description:
 * 
 */

#include <core/node.h>
#include <ops/layer_norm.h>
#include <core/backend/gpu/cuda_common.h>
#include <ops/backend/gpu/impl/layer_norm.h>

namespace mariana {

bool LayerNormFunc::plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (inputs.size() != 1) {
        MLOG(ERROR)<<"LayerNorm input's dimision must be 1";
        return false;
    }
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    cuda_set_device(cuda_ctx->device);
    if (m_weight.device() != DataOn::GPU) {
        m_weight = m_weight.cuda(cuda_ctx->stream());
        m_bias = m_bias.cuda(cuda_ctx->stream());
    }
    Tensor input = inputs[0].shallowcopy();
    if (input.dim_size() == 4) {
        m_owner->runtime_info().feature_height = input.dim_at(1);
        m_owner->runtime_info().feature_width  = input.dim_at(2);
        input.reshape({input.dim_at(0), input.dim_at(1)*input.dim_at(2), input.dim_at(3)});
    }
    if (outputs.empty()) {
        outputs.push_back(Tensor(DataOn::GPU));
    }
    outputs[0].try_realloc(input.dims(), input.dtype());
    return true;
}

bool LayerNormFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    NormParam norm_param = {
        /* .epsilon= */ m_epsilon,
        /* .axies= */ m_axies,
    };
    Tensor input = inputs[0].shallowcopy();
    if (input.dim_size() == 4) {
        input.reshape({input.dim_at(0), input.dim_at(1)*input.dim_at(2), input.dim_at(3)});
    }
    _parallel_sync(m_tp, input.dim_at(0)*input.dim_at(1), layer_normlization, std::ref(input),
                   std::ref(m_weight), std::ref(m_bias), std::ref(norm_param), std::ref(outputs[0]), cuda_ctx);
    return true;
}

} // namespace mariana
