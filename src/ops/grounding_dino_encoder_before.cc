/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/grounding_dino_encoder_before.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-10:09:13:47
 * Description:
 * 
 */

#include <utils/mariana_define.h>
#include <models/model_param.h>
#include <ops/grounding_dino_encoder_before.h>
#include <ops/backend/cpu/grounding_dino_utils.h>

namespace mariana {

bool GroundingDinoEncoderBeforeFunc::init(const ModelParam& param, const std::string& node_name) {
    ModelParam::SafeTensorInfo sti;
    TRY_STL(sti = param.sti_map.at("model.level_embed"), return false);
    Tensor level_embed(sti.shape, DataOn::CPU, sti.data, sti.dtype, true/*move_data*/);
    m_level_embed = level_embed;
    return true;
}

// input dims order is : [N C H W]
bool GroundingDinoEncoderBeforeFunc::plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    size_t level = inputs.size() / 2;
    m_spatial_shapes.size = level;
    int32_t total = 0;
    for (size_t i = 0; i < level; ++i) {
        total += inputs[i].stride_at(1);
        m_spatial_shapes.widths[i] = inputs[i].dim_at(3);
        m_spatial_shapes.heights[i] = inputs[i].dim_at(2);
    }
    context.runtime_info.anything = &m_spatial_shapes;
    if (outputs.empty()) {
        outputs.push_back(Tensor(inputs[0].device()));
        outputs.push_back(Tensor(inputs[0].device()));
    }
    outputs[0].try_realloc({inputs[0].dim_at(0), total, inputs[0].dim_at(1)}, inputs[0].dtype());
    outputs[1].try_realloc({inputs[0].dim_at(0), total, inputs[0].dim_at(1)}, inputs[0].dtype());
    return true;
}

bool GroundingDinoEncoderBeforeFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    TRACE();
    _parallel_sync(m_tp, outputs[0].total_size(), grounding_dino_encoder_before, std::ref(inputs),
                   std::ref(m_level_embed), std::ref(outputs[0]), std::ref(outputs[1]));
    return true;
}

} // namespace mariana
