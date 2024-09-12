/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/grounding_dino_bi_mhs_attention.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-11:09:27:39
 * Description:
 *
 */

#ifndef __OPS_GROUNDING_DINO_BI_MHS_ATTENTION_H__
#define __OPS_GROUNDING_DINO_BI_MHS_ATTENTION_H__

#include <core/function.h>

namespace mariana {

struct SelfAttentionFunc;

struct GroundingDinoBiMHSAttentionFunc : public Function {
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
private:
    int32_t m_attention_head_size = 0;
    int32_t m_n_head              = 0;
    float   m_scale               = 1.f;
    // weights & bias
    Tensor m_vision_proj;
    Tensor m_vision_proj_bias;
    Tensor m_text_proj;
    Tensor m_text_proj_bias;
    Tensor m_vision_value_proj;
    Tensor m_vision_value_proj_bias;
    Tensor m_text_value_proj;
    Tensor m_text_value_proj_bias;
    
    Tensor m_out_vision_proj;
    Tensor m_out_vision_proj_bias;
    
    Tensor m_out_text_proj;
    Tensor m_out_text_proj_bias;
    // outputs
    Tensor m_vision_query_states;
    Tensor m_text_key_states;
    Tensor m_vision_value_states;
    Tensor m_text_value_states;

    Tensor m_vision_value_states_perm;
    Tensor m_text_value_states_perm;
            
    Tensor m_vision_output;
    Tensor m_text_output;
};

} // namespace mariana

#endif /* __OPS_GROUNDING_DINO_BI_MHS_ATTENTION_H__ */

