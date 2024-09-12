/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/grounding_dino_encoder_layer.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-09:17:11:21
 * Description:
 * 
 */

#include <cmath>

#include <models/model_param.h>
#include <utils/mariana_define.h>

#include <core/node.h>
#include <ops/layer_norm.h>
#include <ops/matmul.h>
#include <ops/math.h>
#include <ops/self_attention.h>
#include <ops/grounding_dino_encoder_layer.h>
#include <ops/grounding_dino_encoder_before.h>
#include <ops/grounding_dino_bi_mhs_attention.h>
#include <ops/grounding_dino_multiscale_deformable_attention.h>
#include <ops/backend/cpu/grounding_dino_sine_position_embedding.h>

namespace mariana {

//////////fusion layer//////////////////////////////

GroundingDinoFusionLayerFunc::~GroundingDinoFusionLayerFunc() {
    delete m_layer_norm_vision;
    delete m_layer_norm_text;
    delete m_add_func;
    delete m_bimhs_attn;
}

void GroundingDinoFusionLayerFunc::set_thread_pool(ThreadPool* tp) {
    m_tp = tp;
    m_layer_norm_vision->set_thread_pool(tp);
    m_layer_norm_text->set_thread_pool(tp);
    m_bimhs_attn->set_thread_pool(tp);
    m_add_func->set_thread_pool(tp);
}

bool GroundingDinoFusionLayerFunc::init(const ModelParam& param, const std::string& node_name) {
    m_layer_norm_vision   = new LayerNormFunc{};
    m_layer_norm_text     = new LayerNormFunc{};
    m_add_func            = new AddFunc{};
    m_bimhs_attn          = new GroundingDinoBiMHSAttentionFunc{};
    m_layer_norm_vision->init(param, node_name+".layer_norm_vision");
    m_layer_norm_text->init(param, node_name+".layer_norm_text");
    m_bimhs_attn->init(param, node_name+".attn");
    return true;
}

bool GroundingDinoFusionLayerFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    tensor_list _outputs = {m_ln_vision_out};
    m_layer_norm_vision->plan_forward_cpu({inputs[0]}, _outputs, context);
    _outputs = {m_ln_text_out};
    m_layer_norm_text->plan_forward_cpu({inputs[1]}, _outputs, context);
    m_bimhs_attn->plan_forward_cpu(inputs, outputs, context);
    return true;
}

bool GroundingDinoFusionLayerFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    tensor_list _outputs = {m_ln_vision_out};
    m_layer_norm_vision->on_forward({inputs[0]}, _outputs, context);
    
    _outputs = {m_ln_text_out};
    m_layer_norm_text->on_forward({inputs[1]}, _outputs, context);
    
    tensor_list _inputs = {m_ln_vision_out, m_ln_text_out};
    m_bimhs_attn->on_forward(_inputs, outputs, context);
    
    _inputs = {outputs[0], m_ln_vision_out};
    _outputs = {outputs[0]};
    m_add_func->on_forward(_inputs, _outputs, context);
    
    _inputs = {outputs[2], m_ln_text_out};
    _outputs = {outputs[2]};
    m_add_func->on_forward(_inputs, _outputs, context);
    return true;
}

//////////textenhancer layer//////////////////////////////

GroundingDinoTextEnhancerLayerFunc::~GroundingDinoTextEnhancerLayerFunc() {
    delete m_self_attn;
    delete m_layer_norm_before;
    delete m_layer_norm_after;
    delete m_fc1_func;
    delete m_fc2_func;
    delete m_add_func;
    delete m_sattn_proj;
}

void GroundingDinoTextEnhancerLayerFunc::set_thread_pool(ThreadPool* tp) {
    m_tp = tp;
    m_self_attn->set_thread_pool(tp);
    m_layer_norm_before->set_thread_pool(tp);
    m_layer_norm_after->set_thread_pool(tp);
    m_fc1_func->set_thread_pool(tp);
    m_fc2_func->set_thread_pool(tp);
    m_add_func->set_thread_pool(tp);
    m_sattn_proj->set_thread_pool(tp);
}

bool GroundingDinoTextEnhancerLayerFunc::init(const ModelParam& param, const std::string& node_name) {
    m_num_pos_feats     = param.n_embd/param.n_head;
    m_self_attn         = new SelfAttentionFunc{};
    m_layer_norm_before = new LayerNormFunc{};
    m_layer_norm_after  = new LayerNormFunc{};
    m_fc1_func          = new MatMulFunc{};
    m_fc2_func          = new MatMulFunc{};
    m_add_func          = new AddFunc{};
    m_sattn_proj        = new MatMulFunc{};
    m_layer_norm_before->init(param, node_name+".layer_norm_before");
    m_layer_norm_after->init(param, node_name+".layer_norm_after");
    ModelParam _param = param;
    _param.n_embd /= param.n_head;
    m_self_attn->init(_param, node_name+".self_attn");
    m_sattn_proj->init(param, node_name+".self_attn.out_proj");
    _param.act_cate = OpCategory::RELU;
    m_fc1_func->init(_param, node_name+".fc1");
    m_fc2_func->init(param, node_name+".fc2");
    return true;
}

bool GroundingDinoTextEnhancerLayerFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    m_text_position_embedding.try_realloc({inputs[2].dim_at(0), inputs[2].dim_at(1), m_num_pos_feats}, TypeMeta::make<float>());
    tensor_list _inputs = {inputs[0], m_text_position_embedding};
    tensor_list _outputs = {m_query_key};
    m_add_func->plan_forward_cpu(_inputs, _outputs, context);
    
    _inputs = {m_query_key, m_query_key, inputs[0], inputs[1]};
    _outputs = {m_attention_output};
    m_self_attn->plan_forward_cpu(_inputs, _outputs, context);
    
    _inputs = {m_attention_output};
    _outputs = {m_sattn_proj_output};
    m_sattn_proj->plan_forward_cpu(_inputs, _outputs, context);
    
    _inputs = {m_sattn_proj_output};
    _outputs = {m_ln_before_output};
    m_layer_norm_before->plan_forward_cpu(_inputs, _outputs, context);

    _inputs = {m_ln_before_output};
    _outputs = {m_fc1_output};
    m_fc1_func->plan_forward_cpu(_inputs, _outputs, context);

    _inputs = {m_fc1_output};
    _outputs = {m_fc2_output};
    m_fc2_func->plan_forward_cpu(_inputs, _outputs, context);

    _inputs = {m_fc2_output};
    m_layer_norm_after->plan_forward_cpu(_inputs, outputs, context);
    
    return true;
}

bool GroundingDinoTextEnhancerLayerFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    float scale       = 2*M_PI;
    float temperature = 10000;
    Tensor text_position_ids = inputs[2].shallowcopy();
    text_position_ids.reshape({inputs[2].dim_at(0), inputs[2].dim_at(1), 1});
    _parallel_sync(m_tp, m_text_position_embedding.total_size(), grounding_dino_get_text_enhancer_sine_pos_embed, std::ref(text_position_ids), std::ref(m_text_position_embedding), scale, temperature, false);
    tensor_list _inputs = {inputs[0], m_text_position_embedding};
    tensor_list _outputs = {m_query_key};
    m_add_func->on_forward(_inputs, _outputs, context);
    
    _inputs = {m_query_key, m_query_key, inputs[0], inputs[1]};
    _outputs = {m_attention_output};
    m_self_attn->on_forward(_inputs, _outputs, context);
    
    _inputs = {m_attention_output};
    _outputs = {m_sattn_proj_output};
    m_sattn_proj->on_forward(_inputs, _outputs, context);

    _inputs = {m_sattn_proj_output, inputs[0]};
    _outputs = {m_sattn_proj_output};
    m_add_func->on_forward(_inputs, _outputs, context);

    _inputs = {m_sattn_proj_output};
    _outputs = {m_ln_before_output};
    m_layer_norm_before->on_forward(_inputs, _outputs, context);

    _inputs = {m_ln_before_output};
    _outputs = {m_fc1_output};
    m_fc1_func->on_forward(_inputs, _outputs, context);

    _inputs = {m_fc1_output};
    _outputs = {m_fc2_output};
    m_fc2_func->on_forward(_inputs, _outputs, context);

    _inputs = {m_ln_before_output, m_fc2_output};
    _outputs = {m_fc2_output};
    m_add_func->on_forward(_inputs, _outputs, context);

    _inputs = {m_fc2_output};
    m_layer_norm_after->on_forward(_inputs, outputs, context);
    
    return true;
}

///////////////GroundingDinoDeformableLayerFunc////////////////

GroundingDinoDeformableLayerFunc::~GroundingDinoDeformableLayerFunc() {
    delete m_attn_layer_norm;
    delete m_final_layer_norm;
    delete m_gdmsd_attn_func;
    delete m_fc1_func;
    delete m_fc2_func;
    delete m_add_func;
}

void GroundingDinoDeformableLayerFunc::set_thread_pool(ThreadPool* tp) {
    m_tp = tp;
    m_gdmsd_attn_func->set_thread_pool(tp);
    m_attn_layer_norm->set_thread_pool(tp);
    m_final_layer_norm->set_thread_pool(tp);
    m_fc1_func->set_thread_pool(tp);
    m_fc2_func->set_thread_pool(tp);
    m_add_func->set_thread_pool(tp);
}

bool GroundingDinoDeformableLayerFunc::init(const ModelParam& param, const std::string& node_name) {
    m_attn_layer_norm  = new LayerNormFunc{};
    m_final_layer_norm = new LayerNormFunc{};
    m_gdmsd_attn_func  = new GroundingDinoMultiscaleDeformableAttention{};
    m_fc1_func         = new MatMulFunc{};
    m_fc2_func         = new MatMulFunc{};
    m_add_func         = new AddFunc{};
    m_gdmsd_attn_func->set_node(m_owner);
    m_attn_layer_norm->init(param, node_name+".self_attn_layer_norm");
    m_final_layer_norm->init(param, node_name+".final_layer_norm");
    ModelParam fc1_param = param;
    fc1_param.act_cate = OpCategory::RELU;
    m_fc1_func->init(fc1_param, node_name+".fc1");
    m_fc2_func->init(param, node_name+".fc2");
    m_gdmsd_attn_func->init(param, node_name+".self_attn");
    return true;
}

bool GroundingDinoDeformableLayerFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    tensor_list _inputs = {inputs[0], inputs[0], inputs[1]};
    tensor_list _outputs = {m_gdmsd_attn_t};
    m_gdmsd_attn_func->plan_forward_cpu(_inputs, _outputs, context);

    _outputs = {m_attn_layer_norm_t};
    m_attn_layer_norm->plan_forward_cpu({m_gdmsd_attn_t}, _outputs, context);
    
    _outputs = {m_fc1_t};
    m_fc1_func->plan_forward_cpu({m_attn_layer_norm_t}, _outputs, context);

    m_fc2_func->plan_forward_cpu({m_fc1_t}, outputs, context);
    return true;
}

bool GroundingDinoDeformableLayerFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    tensor_list _inputs = {inputs[0], inputs[0], inputs[1], inputs[2]};
    tensor_list _outputs = {m_gdmsd_attn_t};
    m_gdmsd_attn_func->on_forward(_inputs, _outputs, context);
    
    Tensor residual = inputs[0];
    m_add_func->on_forward({residual, m_gdmsd_attn_t}, _outputs, context);
    _outputs = {m_attn_layer_norm_t};
    m_attn_layer_norm->on_forward({m_gdmsd_attn_t}, _outputs, context);
    
    _outputs = {m_fc1_t};
    m_fc1_func->on_forward({m_attn_layer_norm_t}, _outputs, context);
    
    _outputs = {m_gdmsd_attn_t};
    m_fc2_func->on_forward({m_fc1_t}, _outputs, context);
    residual = m_attn_layer_norm_t;
    
    _outputs = {m_gdmsd_attn_t};
    m_add_func->on_forward({residual, m_gdmsd_attn_t}, _outputs, context);

    m_final_layer_norm->on_forward({m_gdmsd_attn_t}, outputs, context);
    return true;
}

////////encoder layer//////////////////////////////

GroundingDinoEncoderLayerFunc::~GroundingDinoEncoderLayerFunc() {
    delete m_fusion_func;
    delete m_text_enhancer_func;
    delete m_deformable_layer_func;
}

void GroundingDinoEncoderLayerFunc::set_thread_pool(ThreadPool* tp) {
    m_tp = tp;
    m_fusion_func->set_thread_pool(tp);
    m_text_enhancer_func->set_thread_pool(tp);
    m_deformable_layer_func->set_thread_pool(tp);
}

bool GroundingDinoEncoderLayerFunc::init(const ModelParam& param, const std::string& node_name) {
    m_n_levels              = param.num_feature_levels;
    m_fusion_func           = new GroundingDinoFusionLayerFunc{};
    m_text_enhancer_func    = new GroundingDinoTextEnhancerLayerFunc{};
    m_deformable_layer_func = new GroundingDinoDeformableLayerFunc{};
    m_deformable_layer_func->set_node(m_owner);
    m_fusion_func->init(param, node_name+".fusion_layer");
    m_text_enhancer_func->init(param, node_name+".text_enhancer_layer");
    m_deformable_layer_func->init(param, node_name+".deformable_layer");
    return true;
}

void GroundingDinoEncoderLayerFunc::_get_reference_points(ExeContext& context, Tensor& reference_points) {
    GroundingDinoEncoderBeforeFunc::SpatialShapes* sp_shape = static_cast<GroundingDinoEncoderBeforeFunc::SpatialShapes*>(m_owner->info_shared_nodes()[0]->runtime_info().anything);
    uint32_t offset = 0;
    for (uint32_t i = 0; i < sp_shape->size; ++i) {
        for (uint32_t h = 0; h < sp_shape->heights[i]; ++h) {
            for (uint32_t w = 0; w < sp_shape->widths[i]; ++w) {
                for (int32_t j = 0; j < reference_points.dim_at(2); ++j) {
                    *reference_points.unsafe_ptr<float>((offset+h*sp_shape->widths[i]+w)*reference_points.stride_at(1)+j*reference_points.stride_at(2)+0) = (w+0.5f)/sp_shape->widths[i]; // x
                    *reference_points.unsafe_ptr<float>((offset+h*sp_shape->widths[i]+w)*reference_points.stride_at(1)+j*reference_points.stride_at(2)+1) = (h+0.5f)/sp_shape->heights[i];// y
                }
            }
        }
        offset += sp_shape->heights[i]*sp_shape->widths[i];
    }
}

bool GroundingDinoEncoderLayerFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    // inputs' order is [ vision_features, vision_pos_embed, text_features,
    //                    att_mask_pass, position_embedding_pass, reference_points] The outputs' order same as inputs
    if (inputs.size() == 5) {
        m_reference_points.try_realloc({inputs[0].dim_at(0), inputs[0].dim_at(1), m_n_levels, 2}, TypeMeta::make<float>());
        _get_reference_points(context, m_reference_points);
    } else { // inputs.size() == 6
        m_reference_points = inputs[5];
    }
    
    tensor_list _inputs = {inputs[0], inputs[2]};
    tensor_list _outputs = {m_vision_features, m_vision_fused_attn, m_text_features, m_text_fused_attn};
    m_fusion_func->plan_forward_cpu(_inputs, _outputs, context);

    if (outputs.empty()) {
        outputs = {Tensor(inputs[0].device()), inputs[1], Tensor(inputs[0].device()), inputs[3], inputs[4], m_reference_points};
    }

    _inputs = {_outputs[2], inputs[3], inputs[4]};
    _outputs = {outputs[2]};
    m_text_enhancer_func->plan_forward_cpu(_inputs, _outputs, context);
    
    _inputs = {inputs[0], inputs[1]};
    _outputs = {outputs[0]};
    m_deformable_layer_func->plan_forward_cpu(_inputs, _outputs, context);
    return true;
}

bool GroundingDinoEncoderLayerFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    tensor_list _inputs = {inputs[0], inputs[2]};
    tensor_list _outputs = {m_vision_features, m_vision_fused_attn, m_text_features, m_text_fused_attn};
    {
        AUTOTIME("m_fusion_func");
        m_fusion_func->on_forward(_inputs, _outputs, context);
    }
    
    {
        AUTOTIME("m_text_enhancer_func");
        _inputs = {_outputs[2], inputs[3], inputs[4]};
        _outputs = {outputs[2]};
        m_text_enhancer_func->on_forward(_inputs, _outputs, context);
    }
    {
        AUTOTIME("m_deformable_layer_func");
        _inputs = {m_vision_features, inputs[1], m_reference_points};
        _outputs = {outputs[0]};
        m_deformable_layer_func->on_forward(_inputs, _outputs, context);
    }
    return true;
}

} // namespace mariana

