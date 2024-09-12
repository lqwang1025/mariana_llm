/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/matmul.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-06-23:19:19:57
 * Description:
 *
 */

#ifndef __OPS_MATMUL_H__
#define __OPS_MATMUL_H__

#include <ops/ops.h>

#include <core/function.h>

namespace mariana {

struct MatMulFunc : public Function {
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    friend class SwinLayerFunc;
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
private:
    Tensor m_weight;
    Tensor m_bias;
    OpCategory m_act_cate = OpCategory::None;
};

} // namespace mariana

#endif /* __OPS_MATMUL_H__ */

