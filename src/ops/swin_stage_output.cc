/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/swin_stage_output.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-05:15:43:33
 * Description:
 * 
 */

#include <models/model_param.h>
#include <ops/backend/cpu/permute.h>
#include <ops/layer_norm.h>
#include <ops/swin_stage_output.h>

namespace mariana {

SwinStageOutputFunc::~SwinStageOutputFunc() {
    delete m_layer_norm;
}

void SwinStageOutputFunc::set_thread_pool(ThreadPool* tp) {
    m_tp = tp;
    m_layer_norm->set_thread_pool(tp);
}

bool SwinStageOutputFunc::init(const ModelParam& param, const std::string& node_name) {
    m_layer_norm = new LayerNormFunc{};
    m_layer_norm->init(param, node_name);
    return true;
}

bool SwinStageOutputFunc::plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    tensor_list __outputs = {m_ln_out};
    m_layer_norm->plan_forward(inputs, __outputs, context);
    if (outputs.empty()) {
        outputs.push_back(Tensor(inputs[0].device()));
    }
    outputs[0].try_realloc({m_ln_out.dim_at(0), m_ln_out.dim_at(2),
            (int32_t)context.runtime_info.feature_height,
            (int32_t)context.runtime_info.feature_width}, inputs[0].dtype());
    return true;
}

bool SwinStageOutputFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    tensor_list __outputs = {m_ln_out};
    m_layer_norm->on_forward(inputs, __outputs, context);
    uint8_t perms[4] = {0, 3, 1, 2};
    m_ln_out.reshape({m_ln_out.dim_at(0), (int32_t)context.runtime_info.feature_height,
            (int32_t)context.runtime_info.feature_width, m_ln_out.dim_at(2)});
    _parallel_sync(m_tp, m_ln_out.total_size(), permute4, std::ref(m_ln_out), std::ref(outputs[0]), perms);
    return true;
}

} // namespace mariana
