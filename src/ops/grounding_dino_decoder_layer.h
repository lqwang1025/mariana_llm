/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/grounding_dino_decoder_layer.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-08-12:14:49:10
 * Description:
 *
 */

#ifndef __OPS_GROUNDING_DINO_DECODER_LAYER_H__
#define __ops_GROUNDING_DINO_DECODER_LAYER_H__

#include <core/function.h>

namespace mariana {

struct LayerNormFunc;
struct AddFunc;
struct GroundingDinoBiMHSAttentionFunc;
struct SelfAttentionFunc;
struct MatMulFunc;
struct GroundingDinoMultiscaleDeformableAttention;

struct GroundingDinoDecoderLayerFunc : public Function {
    ~GroundingDinoDecoderLayerFunc();
    void set_thread_pool(ThreadPool* tp) override;
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
private:
    int32_t                                     m_d_model                 = 0;
    MatMulFunc*                                 m_reference_points1       = nullptr;
    MatMulFunc*                                 m_reference_points2       = nullptr;
    MatMulFunc*                                 m_sfatt_out_proj          = nullptr;
    MatMulFunc*                                 m_encd_attn_text_out_proj = nullptr;
    MatMulFunc*                                 m_fc1_func                = nullptr;
    MatMulFunc*                                 m_fc2_func                = nullptr;
    AddFunc*                                    m_add_func                = nullptr;
    LayerNormFunc*                              m_attn_ln_func            = nullptr;
    LayerNormFunc*                              m_encd_attn_text_ln_func  = nullptr;
    LayerNormFunc*                              m_encoder_attn_ln_func    = nullptr;
    LayerNormFunc*                              m_final_ln_func           = nullptr;
    SelfAttentionFunc*                          m_self_attn               = nullptr;
    SelfAttentionFunc*                          m_encd_attn_text          = nullptr;
    GroundingDinoMultiscaleDeformableAttention* m_encoder_attn            = nullptr;
    Tensor m_query_pos;
    Tensor m_reference_points1_out;
    Tensor m_qk_out;
    Tensor m_att_out;
    Tensor m_sfatt_out_proj_out;
    Tensor m_attn_ln_out;
    Tensor m_encoder_attn_out;
    Tensor m_fc1_out;
};

} // namespace mariana

#endif /* __ops_GROUNDING_DINO_DECODER_LAYER_H__ */

