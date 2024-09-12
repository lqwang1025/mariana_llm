/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/matmul.cc
 * Authors    : lqwang@pandora
 * Create Time: 2024-06-23:19:21:28
 * Description:
 * 
 */

#include <ops/matmul.h>
#include <ops/backend/cpu/matmul.h>
#include <models/model_param.h>
#include <utils/mariana_define.h>

#include <ops/backend/cpu/matmul.h>

namespace mariana {

bool MatMulFunc::init(const ModelParam& param, const std::string& node_name) {
    TRACE();
    ModelParam::SafeTensorInfo sti;
    TRY_STL(sti = param.sti_map.at(node_name+".weight"), return false);
    Tensor weight(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    m_weight   = weight.deepcopy();
    if (param.sti_map.count(node_name+".bias")) {
        TRY_STL(sti = param.sti_map.at(node_name+".bias"), return false);
        Tensor bias(sti.shape, DataOn::CPU, sti.data, sti.dtype);
        m_bias     = bias.deepcopy();
    }
    m_act_cate = param.act_cate;
    return true;
}

bool MatMulFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    // TODO: SUPPORT other dimension, now support 3 dim only.
    if (outputs.empty()) {
        outputs.push_back(Tensor(inputs[0].device()));
    }
    int32_t nb = inputs[0].dim_at(0);
    int32_t nr = inputs[0].dim_at(1);
    int32_t nc = m_weight.dim_at(0);
    outputs[0].try_realloc({nb, nr, nc}, inputs[0].dtype());
    return true;
}

bool MatMulFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    _parallel_sync(m_tp, inputs[0].dim_at(0)*inputs[0].dim_at(1), matmul, std::ref(inputs[0]),
                   std::ref(m_weight), std::ref(m_bias), std::ref(outputs[0]), 1.f, 1.f, m_act_cate);
    return true;
}

} // namespace mariana
