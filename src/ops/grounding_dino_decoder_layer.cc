/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/grounding_dino_decoder_layer.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-08-12:14:51:22
 * Description:
 * 
 */

#include <cmath>
#include <cfloat>

#include <models/model_param.h>
#include <utils/mariana_define.h>

#include <ops/math.h>
#include <ops/matmul.h>
#include <ops/layer_norm.h>
#include <ops/self_attention.h>
#include <ops/grounding_dino_decoder_layer.h>
#include <ops/grounding_dino_multiscale_deformable_attention.h>
#include <ops/backend/cpu/grounding_dino_sine_position_embedding.h>

namespace mariana {

GroundingDinoDecoderLayerFunc::~GroundingDinoDecoderLayerFunc() {
    delete m_reference_points1;
    delete m_reference_points2;
    delete m_add_func;
    delete m_self_attn;
    delete m_sfatt_out_proj;
    delete m_attn_ln_func;
    delete m_encd_attn_text;
    delete m_encd_attn_text_out_proj;
    delete m_encd_attn_text_ln_func;
    delete m_encoder_attn;
    delete m_encoder_attn_ln_func;
    delete m_fc1_func;
    delete m_fc2_func;
    delete m_final_ln_func;
}

void GroundingDinoDecoderLayerFunc::set_thread_pool(ThreadPool* tp) {
    m_tp = tp;
    m_reference_points1->set_thread_pool(tp);
    m_reference_points2->set_thread_pool(tp);
    m_add_func->set_thread_pool(tp);
    m_self_attn->set_thread_pool(tp);
    m_sfatt_out_proj->set_thread_pool(tp);
    m_attn_ln_func->set_thread_pool(tp);
    m_encd_attn_text->set_thread_pool(tp);
    m_encd_attn_text_out_proj->set_thread_pool(tp);
    m_encd_attn_text_ln_func->set_thread_pool(tp);
    m_encoder_attn->set_thread_pool(tp);
    m_encoder_attn_ln_func->set_thread_pool(tp);
    m_fc1_func->set_thread_pool(tp);
    m_fc2_func->set_thread_pool(tp);
    m_final_ln_func->set_thread_pool(tp);
}

bool GroundingDinoDecoderLayerFunc::init(const ModelParam& param, const std::string& node_name) {
    m_d_model                 = param.d_model;
    m_reference_points1       = new MatMulFunc{};
    m_reference_points2       = new MatMulFunc{};
    m_sfatt_out_proj          = new MatMulFunc{};
    m_add_func                = new AddFunc{};
    m_self_attn               = new SelfAttentionFunc{};
    m_attn_ln_func            = new LayerNormFunc{};
    m_encd_attn_text          = new SelfAttentionFunc{};
    m_encd_attn_text_out_proj = new MatMulFunc{};
    m_encd_attn_text_ln_func  = new LayerNormFunc{};
    m_encoder_attn_ln_func    = new LayerNormFunc{};
    m_encoder_attn            = new GroundingDinoMultiscaleDeformableAttention{};
    m_fc1_func                = new MatMulFunc{};
    m_fc2_func                = new MatMulFunc{};
    m_final_ln_func           = new LayerNormFunc{};
    m_encoder_attn->set_node(m_owner);
    ModelParam _param = param;
    _param.act_cate = OpCategory::RELU;
    m_reference_points1->init(_param, "model.decoder.reference_points_head.layers.0");
    m_reference_points2->init(param, "model.decoder.reference_points_head.layers.1");
    
    m_self_attn->init(_param, node_name+".self_attn");
    m_sfatt_out_proj->init(param, node_name+".self_attn.out_proj");
    m_attn_ln_func->init(param, node_name+".self_attn_layer_norm");
    m_encd_attn_text->init(_param, node_name+".encoder_attn_text");
    m_encd_attn_text_out_proj->init(param, node_name+".encoder_attn_text.out_proj");
    m_encd_attn_text_ln_func->init(param, node_name+".encoder_attn_text_layer_norm");
    m_encoder_attn->init(param, node_name+".encoder_attn");
    m_encoder_attn_ln_func->init(param, node_name+".encoder_attn_layer_norm");
    m_fc1_func->init(_param, node_name+".fc1");
    m_fc2_func->init(param, node_name+".fc2");
    m_final_ln_func->init(param, node_name+".final_layer_norm");
    return true;
}

bool GroundingDinoDecoderLayerFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    Tensor reference_points = inputs[0];
    m_query_pos.try_realloc({reference_points.dim_at(0), reference_points.dim_at(1), reference_points.dim_at(2)*m_d_model/2}, reference_points.dtype());
    tensor_list _outputs = {m_reference_points1_out};
    m_reference_points1->plan_forward_cpu({m_query_pos}, _outputs, context);
    return true;
}

bool GroundingDinoDecoderLayerFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    Tensor reference_points = inputs[0];
    if (reference_points.dim_at(2) == 4) {
        float scale       = 2*M_PI;
        float temperature = 10000;
        _parallel_sync(m_tp, m_query_pos.total_size(), grounding_dino_get_text_enhancer_sine_pos_embed, std::ref(reference_points), std::ref(m_query_pos), scale, temperature, true/*exchange_xy*/);

        tensor_list _outputs = {m_reference_points1_out};
        m_reference_points1->on_forward({m_query_pos}, _outputs, context);

        _outputs = {m_query_pos};
        m_reference_points2->plan_forward_cpu({m_reference_points1_out}, _outputs, context);
        m_reference_points2->on_forward({m_reference_points1_out}, _outputs, context);
        
        Tensor target = inputs[1];
        m_qk_out.try_realloc(m_query_pos.dims(), m_query_pos.dtype());
        _outputs = {m_qk_out};
        m_add_func->on_forward({m_query_pos, target}, _outputs, context);
        _outputs = {m_att_out};
        Tensor __place;
        m_self_attn->plan_forward_cpu({m_qk_out, m_qk_out, target, __place}, _outputs, context);
        m_self_attn->on_forward({m_qk_out, m_qk_out, target, __place}, _outputs, context);

        _outputs = {m_sfatt_out_proj_out};
        m_sfatt_out_proj->plan_forward_cpu({m_att_out}, _outputs, context);
        m_sfatt_out_proj->on_forward({m_att_out}, _outputs, context);

        _outputs = {m_sfatt_out_proj_out};
        m_add_func->on_forward({target, m_sfatt_out_proj_out}, _outputs, context);

        _outputs = {m_attn_ln_out}; // res
        m_attn_ln_func->plan_forward_cpu({m_sfatt_out_proj_out}, _outputs, context);
        m_attn_ln_func->on_forward({m_sfatt_out_proj_out}, _outputs, context);

        _outputs = {m_sfatt_out_proj_out};
        m_add_func->on_forward({m_attn_ln_out, m_query_pos}, _outputs, context);

        Tensor text_encoder_hidden_states = inputs[2];
        
        _outputs = {m_att_out};
        m_encd_attn_text->plan_forward_cpu({m_sfatt_out_proj_out, text_encoder_hidden_states, text_encoder_hidden_states, __place}, _outputs, context);
        m_encd_attn_text->on_forward({m_sfatt_out_proj_out, text_encoder_hidden_states, text_encoder_hidden_states, __place}, _outputs, context);
        
        _outputs = {m_sfatt_out_proj_out};
        m_encd_attn_text_out_proj->plan_forward_cpu({m_att_out}, _outputs, context);
        m_encd_attn_text_out_proj->on_forward({m_att_out}, _outputs, context);

        _outputs = {m_sfatt_out_proj_out};
        m_add_func->on_forward({m_sfatt_out_proj_out, m_attn_ln_out}, _outputs, context);

        _outputs = {m_attn_ln_out};
        m_encd_attn_text_ln_func->plan_forward_cpu({m_sfatt_out_proj_out}, _outputs, context);
        m_encd_attn_text_ln_func->on_forward({m_sfatt_out_proj_out}, _outputs, context);
        // input tensors is : 1. hidden_states 2. encoder_hidden_states 3. position_embeddings 4. reference_point
        Tensor vision_encoder_hidden_states = inputs[3];
        Tensor pos_eming = m_query_pos;
        _outputs = {m_encoder_attn_out};
        m_encoder_attn->plan_forward_cpu({m_attn_ln_out, vision_encoder_hidden_states, pos_eming, reference_points}, _outputs, context);
        m_encoder_attn->on_forward({m_attn_ln_out, vision_encoder_hidden_states, pos_eming, reference_points}, _outputs, context);

        _outputs = {m_encoder_attn_out};
        m_add_func->on_forward({m_encoder_attn_out, m_attn_ln_out}, _outputs, context);
        
        _outputs = {m_attn_ln_out};
        m_encoder_attn_ln_func->plan_forward_cpu({m_encoder_attn_out}, _outputs, context);
        m_encoder_attn_ln_func->on_forward({m_encoder_attn_out}, _outputs, context);

        _outputs = {m_fc1_out};
        m_fc1_func->plan_forward_cpu({m_attn_ln_out}, _outputs, context);
        m_fc1_func->on_forward({m_attn_ln_out}, _outputs, context);

        _outputs = {m_encoder_attn_out};
        m_fc2_func->plan_forward_cpu({m_fc1_out}, _outputs, context);
        m_fc2_func->on_forward({m_fc1_out}, _outputs, context);

        _outputs = {m_attn_ln_out};
        m_add_func->on_forward({m_encoder_attn_out, m_attn_ln_out}, _outputs, context);

        m_final_ln_func->plan_forward_cpu({m_attn_ln_out}, outputs, context);
        m_final_ln_func->on_forward({m_attn_ln_out}, outputs, context);
    } else {
        MLOG(ERROR)<<"GroundingDinoDecoder reference points only support 4 dims.";
        return false;
    }
    return true;
}

} // namespace mariana
