/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/grounding_dino_encoder_layer.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-09:17:09:56
 * Description:
 *
 */

#ifndef __OPS_GROUNDING_DINO_ENCODER_LAYER_H__
#define __OPS_GROUNDING_DINO_ENCODER_LAYER_H__

#include <core/function.h>

namespace mariana {

struct LayerNormFunc;
struct AddFunc;
struct GroundingDinoBiMHSAttentionFunc;
struct SelfAttentionFunc;
struct MatMulFunc;
struct GroundingDinoMultiscaleDeformableAttention;

struct GroundingDinoFusionLayerFunc : public Function {
    ~GroundingDinoFusionLayerFunc();
    void set_thread_pool(ThreadPool* tp) override;
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
private:
    LayerNormFunc*                   m_layer_norm_vision = nullptr;
    LayerNormFunc*                   m_layer_norm_text   = nullptr;
    AddFunc*                         m_add_func          = nullptr;
    GroundingDinoBiMHSAttentionFunc* m_bimhs_attn        = nullptr;
    Tensor                           m_ln_vision_out;
    Tensor                           m_ln_text_out;
};

struct GroundingDinoTextEnhancerLayerFunc : public Function {
    ~GroundingDinoTextEnhancerLayerFunc();
    void set_thread_pool(ThreadPool* tp) override;
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
private:
    bool _get_text_position_embeddings();
private:
    int32_t            m_num_pos_feats     = 1;
    SelfAttentionFunc* m_self_attn         = nullptr;
    LayerNormFunc*     m_layer_norm_before = nullptr;
    LayerNormFunc*     m_layer_norm_after  = nullptr;
    MatMulFunc*        m_fc1_func          = nullptr;
    MatMulFunc*        m_fc2_func          = nullptr;
    AddFunc*           m_add_func          = nullptr;
    MatMulFunc*        m_sattn_proj        = nullptr;
    Tensor m_text_position_embedding;
    Tensor m_query_key;
    Tensor m_attention_output;
    Tensor m_sattn_proj_output;
    Tensor m_ln_before_output;
    Tensor m_ln_after_output;
    Tensor m_fc1_output;
    Tensor m_fc2_output;
};

struct GroundingDinoDeformableLayerFunc : public Function {
    ~GroundingDinoDeformableLayerFunc();
    void set_thread_pool(ThreadPool* tp) override;
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
private:
    GroundingDinoMultiscaleDeformableAttention* m_gdmsd_attn_func = nullptr;
    AddFunc*                                    m_add_func        = nullptr;
    LayerNormFunc*                              m_attn_layer_norm  = nullptr;
    LayerNormFunc*                              m_final_layer_norm = nullptr;
    MatMulFunc*                                 m_fc1_func         = nullptr;
    MatMulFunc*                                 m_fc2_func         = nullptr;
    Tensor                                      m_gdmsd_attn_t;
    Tensor                                      m_attn_layer_norm_t;
    Tensor                                      m_fc1_t;
    Tensor                                      m_fc2_t;
};

struct GroundingDinoEncoderLayerFunc : public Function {
    ~GroundingDinoEncoderLayerFunc();
    void set_thread_pool(ThreadPool* tp) override;
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
private:
    void _get_reference_points(ExeContext& context, Tensor& reference_points);
private:
    int32_t m_n_levels = 0;
    GroundingDinoFusionLayerFunc*       m_fusion_func           = nullptr;
    GroundingDinoTextEnhancerLayerFunc* m_text_enhancer_func    = nullptr;
    GroundingDinoDeformableLayerFunc*   m_deformable_layer_func = nullptr;
    Tensor m_reference_points;
    Tensor m_level_embed;
    Tensor m_vision_features;
    Tensor m_vision_fused_attn;
    Tensor m_text_features;
    Tensor m_text_fused_attn;
};

} // namespace mariana

#endif /* __OPS_GROUNDING_DINO_ENCODER_LAYER_H__ */

