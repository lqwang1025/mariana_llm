/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : permute.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-24:09:24:45
 * Description:
 * 
 */

#include <ops/permute.h>
#include <models/model_param.h>
#include <ops/backend/cpu/permute.h>
#include <utils/mariana_define.h>

namespace mariana {

bool PermuteFunc::init(const ModelParam& param, const std::string& node_name) {
    m_perms[0] = param.perms[0];
    m_perms[1] = param.perms[1];
    m_perms[2] = param.perms[2];
    m_perms[3] = param.perms[3];
    return true;
}

bool PermuteFunc::plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (outputs.empty()) {
        outputs.push_back(Tensor(inputs[0].device()));
    }
    outputs[0].try_realloc({inputs[0].dim_at(m_perms[0]), inputs[0].dim_at(m_perms[1]),
            inputs[0].dim_at(m_perms[2]), inputs[0].dim_at(m_perms[3])}, inputs[0].dtype());
    return true;
}

bool PermuteFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    TRACE();
    _parallel_sync(m_tp, inputs[0].total_size(), permute4, std::ref(inputs[0]), std::ref(outputs[0]), m_perms);
    return true;
}

} // namespace mariana
