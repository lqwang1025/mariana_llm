/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/layer_norm.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-18:13:22:12
 * Description:
 * 
 */

#include <core/node.h>
#include <ops/layer_norm.h>
#include <ops/backend/cpu/normalization.h>
#include <models/model_param.h>
#include <utils/mariana_define.h>

namespace mariana {

bool LayerNormFunc::init(const ModelParam& param, const std::string& node_name) {
    TRACE();
    ModelParam::SafeTensorInfo sti;
    TRY_STL(sti = param.sti_map.at(node_name+".weight"), return false);
    Tensor weight(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    TRY_STL(sti = param.sti_map.at(node_name+".bias"), return false);
    Tensor bias(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    m_weight = weight.deepcopy();
    m_bias = bias.deepcopy();
    m_epsilon = param.layer_norm_eps;
    return true;
}

bool LayerNormFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    Tensor input = inputs[0].shallowcopy();
    if (input.dim_size() == 4) {
        m_owner->runtime_info().feature_height = input.dim_at(1);
        m_owner->runtime_info().feature_width  = input.dim_at(2);
        input.reshape({input.dim_at(0), input.dim_at(1)*input.dim_at(2), input.dim_at(3)});
    }
    if (outputs.empty()) {
        outputs.push_back(Tensor(input.device()));
    }
    outputs[0].try_realloc(input.dims(), input.dtype());
    return true;
}

// Just support 3 dim now!
bool LayerNormFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    TRACE();
    NormParam norm_param = {
        /* .epsilon= */ m_epsilon,
        /* .axies= */ m_axies,
    };
    Tensor input = inputs[0].shallowcopy();
    if (input.dim_size() == 4) {
        input.reshape({input.dim_at(0), input.dim_at(1)*input.dim_at(2), input.dim_at(3)});
    }
    _parallel_sync(m_tp, input.dim_at(0)*input.dim_at(1), layer_normlization, std::ref(input),
                   std::ref(m_weight), std::ref(m_bias), std::ref(outputs[0]), std::ref(norm_param));
    return true;
}

} // namespace mariana
