/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : rms_norm.cc
 * Authors    : lqwang@SMT23090002
 * Create Time: 2025-03-21:11:49:01
 * Description:
 * 
 */

#include <core/node.h>
#include <ops/rms_norm.h>
// #include <ops/backend/cpu/normalization.h>
#include <models/model_param.h>
#include <utils/mariana_define.h>

namespace mariana {

bool RMSNormFunc::init(const ModelParam& param, const std::string& node_name) {
    TRACE();
    ModelParam::SafeTensorInfo sti;
    TRY_STL(sti = param.sti_map.at(node_name+".weight"), return false);
    Tensor weight(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    m_weight = weight.deepcopy();
    m_epsilon = param.layer_norm_eps;
    return true;
}

bool RMSNormFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    MLOG(FATAL)<<"CPU RMSNorm TODO";
    return true;
}

bool RMSNormFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    MLOG(FATAL)<<"CPU RMSNorm TODO";
    return true;
}

} // namespace mariana
