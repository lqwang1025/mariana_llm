/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/grounding_dino_encoder_before.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-15:14:35:00
 * Description:
 *
 */

#include <core/node.h>
#include <core/backend/gpu/cuda_common.h>

#include <ops/grounding_dino_encoder_before.h>
#include <ops/backend/gpu/impl/grounding_dino_utils.h>

namespace mariana {

bool GroundingDinoEncoderBeforeFunc::plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    cuda_set_device(cuda_ctx->device);
    if (m_level_embed.device() != DataOn::GPU) {
        m_level_embed = m_level_embed.cuda(cuda_ctx->stream());
    }
    size_t level = inputs.size() / 2;
    m_spatial_shapes.size = level;
    int32_t total = 0;
    for (size_t i = 0; i < level; ++i) {
        total += inputs[i].stride_at(1);
        m_spatial_shapes.widths[i] = inputs[i].dim_at(3);
        m_spatial_shapes.heights[i] = inputs[i].dim_at(2);
    }
    m_owner->runtime_info().anything = &m_spatial_shapes;
    if (outputs.empty()) {
        outputs.push_back(Tensor(DataOn::GPU));
        outputs.push_back(Tensor(DataOn::GPU));
    }
    outputs[0].try_realloc({inputs[0].dim_at(0), total, inputs[0].dim_at(1)}, inputs[0].dtype());
    outputs[1].try_realloc({inputs[0].dim_at(0), total, inputs[0].dim_at(1)}, inputs[0].dtype());
    return true;
}

bool GroundingDinoEncoderBeforeFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    _parallel_sync(m_tp, 1, grounding_dino_encoder_before, std::ref(inputs),
                   std::ref(m_level_embed), std::ref(outputs[0]), std::ref(outputs[1]), cuda_ctx);
    return true;    
}

} // namespace mariana
