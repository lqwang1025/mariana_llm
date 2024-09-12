/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/group_norm.cc
 * Authors    : lqwang@pandora
 * Create Time: 2024-07-06:09:29:26
 * Description:
 * 
 */

#include <ops/group_norm.h>
#include <ops/layer_norm.h>
#include <ops/backend/cpu/normalization.h>
#include <models/model_param.h>
#include <utils/mariana_define.h>
#include <core/tensor_utils.h>

namespace mariana {

bool GroupNormFunc::init(const ModelParam& param, const std::string& node_name) {
    TRACE();
    ModelParam::SafeTensorInfo sti;
    TRY_STL(sti = param.sti_map.at(node_name+".weight"), return false);
    Tensor weight(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    TRY_STL(sti = param.sti_map.at(node_name+".bias"), return false);
    Tensor bias(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    m_weight  = weight.deepcopy();
    m_bias    = bias.deepcopy();
    m_epsilon = param.layer_norm_eps;
    m_group   = static_cast<uint8_t>(param.groups);
    return true;
}

bool GroupNormFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    Tensor input = inputs[0];
    if (outputs.empty()) {
        outputs.push_back(Tensor(input.device()));
    }
    outputs[0].try_realloc(input.dims(), input.dtype());
    return true;
}

// Just support 3 dim now!
bool GroupNormFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    TRACE();
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
        _parallel_sync(m_tp, input.dim_at(0)*input.dim_at(1), group_normlization, std::ref(input), std::ref(m_weight), std::ref(m_bias), std::ref(outputs[0]), std::ref(norm_param));
        return true;
    } else {
        MLOG(ERROR)<<"Unsupport dim number now :"<<input.dim_size();
        return false;
    }
}

} // namespace mariana
