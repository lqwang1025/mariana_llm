/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : gelu.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-25:16:51:09
 * Description:
 * 
 */

#include <ops/gelu.h>
#include <ops/backend/cpu/gelu.h>

#include <utils/mariana_define.h>

namespace mariana {

bool GELUFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (outputs.empty()) {
        outputs.push_back(Tensor(inputs[0].device()));
    }
    outputs[0].try_realloc(inputs[0].dims(), inputs[0].dtype());
    return true;
}

bool GELUFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    _parallel_sync(m_tp, inputs[0].total_size(), gelu, std::ref(inputs[0]), std::ref(outputs[0]));
    return true;
}

} // namespace mariana

