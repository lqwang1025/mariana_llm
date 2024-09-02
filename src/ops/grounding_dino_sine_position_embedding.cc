/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/grounding_dino_sine_position_embedding.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-08:10:03:07
 * Description:
 * 
 */

#include <utils/mariana_define.h>
#include <models/model_param.h>
#include <ops/grounding_dino_sine_position_embedding.h>
#include <ops/backend/cpu/grounding_dino_sine_position_embedding.h>

namespace mariana {

bool GroundingDinoSinePositionEmbeddingFunc::init(const ModelParam& param, const std::string& node_name) {   
    m_temperature   = param.positional_embedding_temperature;
    return true;
}

bool GroundingDinoSinePositionEmbeddingFunc::plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (outputs.empty()) {
        outputs.push_back(Tensor(inputs[0].device()));
    }
    outputs[0].try_realloc(inputs[0].dims(), TypeMeta::make<float>());
    return true;
}

bool GroundingDinoSinePositionEmbeddingFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    TRACE();
    _parallel_sync(m_tp, outputs[0].total_size(), grounding_dino_sine_position_embedding, std::ref(inputs[0]), std::ref(outputs[0]), m_scale, m_temperature);
    return true;
}

} // namespace mariana

