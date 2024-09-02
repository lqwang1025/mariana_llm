/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/grounding_dino_multiscale_deformable_attention.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-07-21:07:54:24
 * Description:
 *
 */

#ifndef __GROUNDING_DINO_MULTISCALE_DEFORMABLE_ATTENTION_H__
#define __GROUNDING_DINO_MULTISCALE_DEFORMABLE_ATTENTION_H__

#include <core/function.h>

namespace mariana {

struct AddFunc;

struct GroundingDinoMultiscaleDeformableAttention : public Function {
    ~GroundingDinoMultiscaleDeformableAttention();
    void set_thread_pool(ThreadPool* tp) override;
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
private:
    int32_t m_d_model  = 0;
    int32_t m_n_levels = 0;
    int32_t m_n_heads  = 0;
    int32_t m_n_points = 0;
    AddFunc* m_add_func = nullptr;
    Tensor m_value_proj_weight;
    Tensor m_value_proj_bias;
    Tensor m_sampling_offsets_weight;
    Tensor m_sampling_offsets_bias;
    Tensor m_attention_weights_weight;
    Tensor m_attention_weights_bias;
    Tensor m_output_proj_weight;
    Tensor m_output_proj_bias;
    // output
    Tensor m_add_out;
    Tensor m_value_out;
    Tensor m_sampling_offsets_out;
    Tensor m_attention_weights_out;
    Tensor m_attention_weights_softmax_out;
    Tensor m_msda_out;
};

} // namespace mariana 

#endif /* __GROUNDING_DINO_MULTISCALE_DEFORMABLE_ATTENTION_H__ */

