/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/math.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-21:18:25:28
 * Description:
 * 
 */

#include <ops/math.h>
#include <ops/backend/cpu/add.h>
#include <ops/backend/cpu/mul.h>
#include <utils/mariana_define.h>

namespace mariana {

bool AddFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (outputs.empty()) {
        outputs.push_back(Tensor(inputs[0].device()));
    }
    outputs[0].try_realloc(inputs[0].dims(), inputs[0].dtype());
    return true;
}

bool AddFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    TRACE();
    _parallel_sync(m_tp, inputs[0].total_size(), add_ele, std::ref(inputs[0]),
                   std::ref(inputs[1]), std::ref(outputs[0]));
    return true;
}

bool MulFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (outputs.empty()) {
        outputs.push_back(Tensor(inputs[0].device()));
    }
    outputs[0].try_realloc(inputs[0].dims(), inputs[0].dtype());
    return true;
}

bool MulFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    TRACE();
    if (inputs[0].total_size() == inputs[1].total_size()) {
        _parallel_sync(m_tp, inputs[0].total_size(), mul_ele, std::ref(inputs[0]),
                       std::ref(inputs[1]), std::ref(outputs[0]));
    } else if (inputs[0].dim_size() == 2 && inputs[1].dim_size() == 1) {
        if (inputs[0].dim_at(0) != inputs[1].dim_at(0) &&
            inputs[0].dim_at(1) != inputs[1].dim_at(0)) {
            MLOG(ERROR)<<"Mul unsupport brodacast size!!";
            return false;
        }
        _parallel_sync(m_tp, inputs[0].total_size(), broadcast_mul2, std::ref(inputs[0]),
                       std::ref(inputs[1]), std::ref(outputs[0]));
    } else {
        MLOG(ERROR)<<"Mul unsupport dim size:"<<inputs[0].dim_size();
        return false;
    }
    
    return true;
}

} // namespace mariana
