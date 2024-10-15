/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/group_norm.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-15:05:52:51
 * Description:
 * 
 */
#include <core/node.h>
#include <core/backend/gpu/cuda_common.h>

#include <ops/layer_norm.h>
#include <ops/group_norm.h>
#include <ops/backend/gpu/impl/layer_norm.h>

namespace mariana {

bool GroupNormFunc::plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (inputs.size() != 1) {
        MLOG(ERROR)<<"GroupNorm input's size must be 1 now:"<<inputs.size();
        return false;
    }
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    cuda_set_device(cuda_ctx->device);
    if (m_weight.device() != DataOn::GPU) {
        m_weight = m_weight.cuda(cuda_ctx->stream());
        m_bias = m_bias.cuda(cuda_ctx->stream());
    }
    Tensor input = inputs[0];
    if (outputs.empty()) {
        outputs.push_back(Tensor(DataOn::GPU));
    }
    outputs[0].try_realloc(input.dims(), input.dtype());
    return true;
}

bool GroupNormFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    NormParam norm_param = {
        /* .epsilon= */ m_epsilon,
        /* .axies= */ m_axies,
        /* .groups= */ m_group,
    };
    Tensor input = inputs[0].shallowcopy();
    if (input.dim_size() == 4) {
        if (input.dim_at(1)%m_group != 0) {
            MLOG(ERROR)<<"Group must be divided by channels";
            return false;
        }
        input.reshape({input.dim_at(0), m_group, input.dim_at(1)/m_group, input.dim_at(2)*input.dim_at(3)});
        _parallel_sync(m_tp, input.dim_at(0), group_normlization, std::ref(input), std::ref(m_weight), std::ref(m_bias), std::ref(outputs[0]), std::ref(norm_param), cuda_ctx);
        return true;
    } else {
        MLOG(ERROR)<<"Unsupport dim number now :"<<input.dim_size();
        return false;
    }
    return true;
}

}
