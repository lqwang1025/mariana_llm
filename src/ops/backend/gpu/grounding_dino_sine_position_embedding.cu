/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : grounding_dino_sine_position_embedding.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-15:09:25:52
 * Description:
 * 
 */

#include <ops/grounding_dino_sine_position_embedding.h>
#include <ops/backend/gpu/impl/grounding_dino_sine_position_embedding.h>

#include <core/node.h>
#include <core/backend/gpu/cuda_common.h>

namespace mariana {

bool GroundingDinoSinePositionEmbeddingFunc::plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    cuda_set_device(cuda_ctx->device);
    if (outputs.empty()) {
        outputs.push_back(Tensor(DataOn::GPU));
    }
    outputs[0].try_realloc(inputs[0].dims(), TypeMeta::make<float>());
    return true;
}

bool GroundingDinoSinePositionEmbeddingFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    _parallel_sync(m_tp, 1, grounding_dino_sine_position_embedding, std::ref(inputs[0]), std::ref(outputs[0]), m_scale, m_temperature, cuda_ctx);
    return true;
}

} // namespace mariana
