/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/roll.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-03:09:03:44
 * Description:
 * 
 */

#include <ops/roll.h>
#include <models/model_param.h>
#include <utils/mariana_define.h>
#include <ops/backend/cpu/roll.h>

namespace mariana {

bool RollFunc::init(const ModelParam& param, const std::string& node_name) {
    MLOG(FATAL)<<"TODO: RollFunc is internal operator!!!";
    return true;
}

bool RollFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (outputs.empty()) {
        outputs.push_back(Tensor(inputs[0].device()));
    }
    outputs[0].try_realloc(inputs[0].dims(), inputs[0].dtype());
    return true;
}

bool RollFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    TRACE();
    _parallel_sync(m_tp, outputs[0].total_size(), roll4, std::ref(inputs[0]), std::ref(outputs[0]), param);
    return true;    
}

} // namespace mariana
