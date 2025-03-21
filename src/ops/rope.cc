/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/rope.cc
 * Authors    : lqwang@SMT23090002
 * Create Time: 2025-03-17:09:08:59
 * Description:
 * 
 */

#include <cmath>
#include <ops/rope.h>
#include <models/model_param.h>
#include <utils/mariana_define.h>
// #include <ops/backend/cpu/roll.h>

namespace mariana {

Tensor ROPEFunc::_compute_default_rope_parameters(const ModelParam& model_param) {
    int32_t head_dim = static_cast<int32_t>(model_param.n_embd/model_param.n_head);
    Tensor ret({1, head_dim/2});
    int32_t count = 0;
    for (int32_t i = 0; i < head_dim; i+=2) {
        float item = 1.f / pow(this->param.rope_theta, ((float)i/(float)head_dim));
        ret.mutable_ptr<float>()[count++] = item;
    }
    param.attention_factor = 1.f;
    return ret;
}

bool ROPEFunc::init(const ModelParam& model_param, const std::string& node_name) {
    this->param.rope_theta = model_param.rope_theta;
    this->param.partial_rotary_factor = model_param.partial_rotary_factor;
    this->param.rope_type = model_param.rope_type;
    if (this->param.rope_type == "default") {
        param.inv_freq = _compute_default_rope_parameters(model_param);
    } else {
        MLOG(FATAL)<<"ROPE unsupport type:"<<this->param.rope_type;
    }
    return true;
}

bool ROPEFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    MLOG(FATAL)<<"CPU TODO";
    return true;
}

bool ROPEFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    TRACE();
    MLOG(FATAL)<<"CPU TODO";
    // _parallel_sync(m_tp, outputs[0].total_size(), roll4, std::ref(inputs[0]), std::ref(outputs[0]), param);
    return true;    
}

} // namespace mariana
