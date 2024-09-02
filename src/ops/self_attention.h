/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/self_attention.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-06-23:15:47:21
 * Description:
 *
 */

#ifndef __OPS_SELF_ATTENTION_H__
#define __OPS_SELF_ATTENTION_H__

#include <core/tensor.h>
#include <core/function.h>

namespace mariana {

struct SelfAttentionFunc : public Function {
    virtual ~SelfAttentionFunc() {}
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    friend class SwinLayerFunc;
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    int32_t m_attention_head_size = 0;
    int32_t m_n_head = 0;
    Tensor m_q_weight;
    Tensor m_q_bias;
    Tensor m_k_weight;
    Tensor m_k_bias;
    Tensor m_v_weight;
    Tensor m_v_bias;
    
    Tensor m_q_o;
    Tensor m_k_o;
    Tensor m_v_o;
};

} // namespace mariana

#endif /* __OPS_SELF_ATTENTION_H__ */

