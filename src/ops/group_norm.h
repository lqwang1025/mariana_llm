/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/group_norm.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-07-06:09:25:35
 * Description:
 *
 */

#ifndef __OPS_GROUP_NORM_H__
#define __OPS_GROUP_NORM_H__

#include <core/function.h>

namespace mariana {

struct GroupNormFunc : public Function {
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
private:
    Tensor m_weight;
    Tensor m_bias;
    float m_epsilon = 1e-5;
    int8_t m_axies = -1;
    uint8_t m_group = 1;
};

} // namespace mariana

#endif /* __OPS_GROUP_NORM_H__ */

