/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/grounding_dino_decoder_before.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-08-14:14:35:48
 * Description:
 *
 */

#ifndef __OPS_GROUNDING_DINO_DECODER_BEFORE_H__
#define __OPS_GROUNDING_DINO_DECODER_BEFORE_H__

#include <vector>

#include <core/function.h>

namespace mariana {

struct AddFunc;
struct LayerNormFunc;
struct MatMulFunc;
struct GroundingDinoDecoderLayerFunc;

struct GroundingDinoDecoderBeforeFunc : public Function {
    ~GroundingDinoDecoderBeforeFunc();
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
    void set_thread_pool(ThreadPool* tp) override;
protected:
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
private:
    int32_t        m_decoder_layers              = 0;
    int32_t        m_n_levels                    = 0;
    int32_t        m_n_queries                   = 0;
    LayerNormFunc* m_enc_output_norm_func        = nullptr;
    LayerNormFunc* m_hidden_stats_ln_func        = nullptr;
    MatMulFunc*    m_enc_output_class_embed_func = nullptr;
    MatMulFunc*    m_enc_output_bbox_embed_func0 = nullptr;
    MatMulFunc*    m_enc_output_bbox_embed_func1 = nullptr;
    MatMulFunc*    m_enc_output_bbox_embed_func2 = nullptr;
    
    MatMulFunc*    m_dec_output_bbox_embed_func0 = nullptr;
    MatMulFunc*    m_dec_output_bbox_embed_func1 = nullptr;
    MatMulFunc*    m_dec_output_bbox_embed_func2 = nullptr;
        
    AddFunc*       m_add_func                    = nullptr;
    std::vector<GroundingDinoDecoderLayerFunc*> m_gd_decoder_layers;
    // output tensor
    Tensor m_output_proposals;
    Tensor m_object_query;
    Tensor m_enc_output;
    Tensor m_enc_output_norm;
    Tensor m_vt_output;
    
    Tensor m_enc_output_bbox_embed0_output;
    Tensor m_enc_output_bbox_embed1_output;
    Tensor m_enc_outputs_coord_logits;
    Tensor m_topk_logits;
    Tensor m_topk_indices;
    Tensor m_query_embeds;
    Tensor m_topk_coords_logits;
    Tensor m_decoder_out;
};

} // namespace mariana

#endif /* __OPS_GROUNDING_DINO_DECODER_BEFORE_H__ */

