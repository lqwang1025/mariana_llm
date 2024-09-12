/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/grounding_dino_decoder_before.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-08-14:14:36:29
 * Description:
 * 
 */

#include <cmath>
#include <cfloat>

#include <core/node.h>
#include <core/tensor_utils.h>

#include <core/data_type.h>
#include <models/model_param.h>
#include <utils/mariana_define.h>

#include <ops/backend/cpu/get_rows.h>
#include <ops/layer_norm.h>
#include <ops/matmul.h>
#include <ops/math.h>
#include <ops/backend/cpu/matmul.h>
#include <ops/backend/cpu/max.h>
#include <ops/backend/cpu/sigmoid.h>
#include <ops/grounding_dino_decoder_layer.h>
#include <ops/grounding_dino_encoder_before.h>
#include <ops/grounding_dino_decoder_before.h>
#include <ops/backend/cpu/grounding_dino_utils.h>

#include <absl/strings/str_format.h>

namespace mariana {

bool GroundingDinoDecoderBeforeFunc::init(const ModelParam& param, const std::string& node_name) {
    m_n_levels                    = param.num_feature_levels;
    m_n_queries                   = param.num_queries;
    m_enc_output_class_embed_func = new MatMulFunc{};
    m_enc_output_bbox_embed_func0 = new MatMulFunc{};
    m_enc_output_bbox_embed_func1 = new MatMulFunc{};
    m_enc_output_bbox_embed_func2 = new MatMulFunc{};
    
    m_dec_output_bbox_embed_func0 = new MatMulFunc{};
    m_dec_output_bbox_embed_func1 = new MatMulFunc{};
    m_dec_output_bbox_embed_func2 = new MatMulFunc{};
    m_enc_output_norm_func        = new LayerNormFunc{};
    m_hidden_stats_ln_func        = new LayerNormFunc{};
    m_add_func                    = new AddFunc{};
    m_enc_output_class_embed_func->init(param, "model.enc_output");
    m_enc_output_norm_func->init(param, "model.enc_output_norm");

    m_hidden_stats_ln_func->init(param, "model.decoder.layer_norm");
    
    ModelParam relu_param = param;
    relu_param.act_cate = OpCategory::RELU;
    m_enc_output_bbox_embed_func0->init(relu_param, "model.encoder_output_bbox_embed.layers.0");
    m_enc_output_bbox_embed_func1->init(relu_param, "model.encoder_output_bbox_embed.layers.1");
    m_enc_output_bbox_embed_func2->init(param, "model.encoder_output_bbox_embed.layers.2");

    if (true) { // TODO: check decoder_bbox_embed_share is true
        m_dec_output_bbox_embed_func0->init(relu_param, "bbox_embed.0.layers.0");
        m_dec_output_bbox_embed_func1->init(relu_param, "bbox_embed.0.layers.1");
        m_dec_output_bbox_embed_func2->init(param, "bbox_embed.0.layers.2");
    } else {
        return false;
    }
        
    ModelParam::SafeTensorInfo sti;
    TRY_STL(sti = param.sti_map.at("model.query_position_embeddings.weight"), return false);
    Tensor query_embeds(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    m_query_embeds = query_embeds.deepcopy();
    if (m_query_embeds.dim_size() == 2) {
        m_query_embeds.reshape({1, m_query_embeds.dim_at(0), m_query_embeds.dim_at(1)});
    }
    m_decoder_layers = param.decoder_layers;
    for (int32_t i = 0; i < param.decoder_layers; ++i) {
        GroundingDinoDecoderLayerFunc* layer = new GroundingDinoDecoderLayerFunc{};
        layer->set_node(m_owner);
        std::string name = absl::StrFormat("model.decoder.layers.%d", i);
        layer->init(param, name);
        m_gd_decoder_layers.push_back(layer);
    }
    
    return true;
}

GroundingDinoDecoderBeforeFunc::~GroundingDinoDecoderBeforeFunc() {
    delete m_enc_output_class_embed_func;
    delete m_enc_output_norm_func;
    delete m_enc_output_bbox_embed_func0;
    delete m_enc_output_bbox_embed_func1;
    delete m_enc_output_bbox_embed_func2;
    delete m_dec_output_bbox_embed_func0;
    delete m_dec_output_bbox_embed_func1;
    delete m_dec_output_bbox_embed_func2;
    delete m_hidden_stats_ln_func;
    delete m_add_func;
    for (auto layer : m_gd_decoder_layers) {
        delete layer;
    }
    m_gd_decoder_layers.clear();
}

void GroundingDinoDecoderBeforeFunc::set_thread_pool(ThreadPool* tp) {
    m_tp = tp;
    m_enc_output_class_embed_func->set_thread_pool(tp);
    m_enc_output_norm_func->set_thread_pool(tp);
    m_enc_output_bbox_embed_func0->set_thread_pool(tp);
    m_enc_output_bbox_embed_func1->set_thread_pool(tp);
    m_enc_output_bbox_embed_func2->set_thread_pool(tp);

    m_dec_output_bbox_embed_func0->set_thread_pool(tp);
    m_dec_output_bbox_embed_func1->set_thread_pool(tp);
    m_dec_output_bbox_embed_func2->set_thread_pool(tp);
    m_add_func->set_thread_pool(tp);
    m_hidden_stats_ln_func->set_thread_pool(tp);
    for (auto layer : m_gd_decoder_layers) {
        layer->set_thread_pool(tp);
    }
}

bool GroundingDinoDecoderBeforeFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    m_output_proposals.try_realloc({inputs[0].dim_at(0), inputs[0].dim_at(1), m_n_levels}, inputs[0].dtype());
    m_object_query.try_realloc(inputs[0].dims(), inputs[0].dtype());
    tensor_list _outputs = {m_enc_output};
    m_enc_output_class_embed_func->plan_forward_cpu({m_object_query}, _outputs, context);
    
    _outputs = {m_enc_output_norm};
    m_enc_output_norm_func->plan_forward_cpu({m_enc_output}, _outputs, context);

    _outputs = {m_enc_output_bbox_embed0_output};
    m_enc_output_bbox_embed_func0->plan_forward_cpu({m_enc_output_norm}, _outputs, context);

    _outputs = {m_enc_output_bbox_embed1_output};
    m_enc_output_bbox_embed_func1->plan_forward_cpu({m_enc_output_bbox_embed0_output}, _outputs, context);

    _outputs = {m_enc_outputs_coord_logits};
    m_enc_output_bbox_embed_func2->plan_forward_cpu({m_enc_output_bbox_embed1_output}, _outputs, context);

    m_vt_output.try_realloc({m_enc_output_norm.dim_at(0), m_enc_output_norm.dim_at(1), inputs[2].dim_at(1)}, inputs[0].dtype());

    m_topk_logits.try_realloc({m_vt_output.dim_at(0), m_vt_output.dim_at(1)}, m_vt_output.dtype());
    m_topk_indices.try_realloc({m_topk_logits.dim_at(0), m_n_queries}, TypeMeta::make<int32_t>());
    
    m_topk_coords_logits.try_realloc({m_enc_outputs_coord_logits.dim_at(0), m_n_queries, m_enc_outputs_coord_logits.dim_at(2)}, m_enc_outputs_coord_logits.dtype());
    Tensor reference_points = m_topk_coords_logits;
    for (auto layer : m_gd_decoder_layers) {
        _outputs = {};
        layer->plan_forward_cpu({reference_points, m_query_embeds, inputs[2], inputs[0]}, _outputs, context);
    }
    return true;
}

bool GroundingDinoDecoderBeforeFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    GroundingDinoEncoderBeforeFunc::SpatialShapes* sp_shape = static_cast<GroundingDinoEncoderBeforeFunc::SpatialShapes*>(m_owner->info_shared_nodes()[0]->runtime_info().anything);
    _parallel_sync(m_tp, inputs[0].dim_at(1), generate_encoder_output_proposals, std::ref(*sp_shape), std::ref(inputs[0]), std::ref(m_output_proposals), std::ref(m_object_query));
    tensor_list _outputs = {m_enc_output};
    m_enc_output_class_embed_func->on_forward({m_object_query}, _outputs, context);
    
    _outputs = {m_enc_output_norm};
    m_enc_output_norm_func->on_forward({m_enc_output}, _outputs, context);
    
    Tensor weight = inputs[2].shallowcopy();
    weight.reshape({inputs[2].dim_at(0)*inputs[2].dim_at(1), inputs[2].dim_at(2)});
    Tensor __bias;
    _parallel_async(m_tp, m_enc_output_norm.dim_at(0)*m_enc_output_norm.dim_at(1), matmul, std::ref(m_enc_output_norm), std::ref(weight), std::ref(__bias), std::ref(m_vt_output), 1.f, 1.f, OpCategory::None);
    
    // encoder_output_bbox_embed
    _outputs = {m_enc_output_bbox_embed0_output};
    m_enc_output_bbox_embed_func0->on_forward({m_enc_output_norm}, _outputs, context);
    
    _outputs = {m_enc_output_bbox_embed1_output};
    m_enc_output_bbox_embed_func1->on_forward({m_enc_output_bbox_embed0_output}, _outputs, context);

    _outputs = {m_enc_outputs_coord_logits};
    m_enc_output_bbox_embed_func2->on_forward({m_enc_output_bbox_embed1_output}, _outputs, context);
    m_tp->wait_work_complete();

    _parallel_async(m_tp, m_vt_output.dim_at(0)*m_vt_output.dim_at(1), max_last_dim_spilt, std::ref(m_vt_output), std::ref(m_topk_logits));
    
    _outputs = {m_enc_outputs_coord_logits};
    m_add_func->on_forward({m_output_proposals, m_enc_outputs_coord_logits}, _outputs, context);
    m_tp->wait_work_complete();
    topk_index(m_n_queries, m_topk_logits, m_topk_indices);

    m_enc_outputs_coord_logits.reshape({m_enc_outputs_coord_logits.dim_at(0)*m_enc_outputs_coord_logits.dim_at(1), m_enc_outputs_coord_logits.dim_at(2)});

    _parallel_sync(m_tp, m_topk_indices.total_size(), get_rows, std::ref(m_topk_indices),
                   std::ref(m_enc_outputs_coord_logits), std::ref(m_topk_coords_logits));

    _parallel_sync(m_tp, m_topk_coords_logits.total_size(), sigmoid, std::ref(m_topk_coords_logits),
                   std::ref(m_topk_coords_logits));
    if (outputs.empty()) {
        outputs.resize(2*m_decoder_layers+1);
    }
    outputs[1] = m_topk_coords_logits.deepcopy();
    outputs[2*m_decoder_layers] = inputs[2].deepcopy();
    
    Tensor reference_points = m_topk_coords_logits;
    Tensor hidden_states    = m_query_embeds;
    int32_t i = 0;
    for (auto layer : m_gd_decoder_layers) {
        _outputs = {m_decoder_out};
        layer->on_forward({reference_points, hidden_states, inputs[2], inputs[0]}, _outputs, context);
        hidden_states = m_decoder_out;
        
        _outputs = {outputs[2*i+0]};
        m_hidden_stats_ln_func->plan_forward_cpu({hidden_states}, _outputs, context);
        m_hidden_stats_ln_func->on_forward({hidden_states}, _outputs, context);
        
        _outputs = {m_enc_output_bbox_embed0_output};
        m_dec_output_bbox_embed_func0->plan_forward_cpu({hidden_states}, _outputs, context);
        m_dec_output_bbox_embed_func0->on_forward({hidden_states}, _outputs, context);
        
        _outputs = {m_enc_output_bbox_embed1_output};
        m_dec_output_bbox_embed_func1->plan_forward_cpu({m_enc_output_bbox_embed0_output}, _outputs, context);
        m_dec_output_bbox_embed_func1->on_forward({m_enc_output_bbox_embed0_output}, _outputs, context);

        _outputs = {m_enc_outputs_coord_logits};
        m_dec_output_bbox_embed_func2->plan_forward_cpu({m_enc_output_bbox_embed1_output}, _outputs, context);
        m_dec_output_bbox_embed_func2->on_forward({m_enc_output_bbox_embed1_output}, _outputs, context);
        if (i != m_decoder_layers-1) {
            outputs[2*(i+1)+1].try_realloc(m_topk_coords_logits.dims(), m_topk_coords_logits.dtype());
            _parallel_sync(m_tp, m_enc_outputs_coord_logits.total_size(), decoder_reference_points_correct, std::ref(m_enc_outputs_coord_logits), std::ref(reference_points), std::ref(outputs[2*(i+1)+1]), 1e-5);
            reference_points = outputs[2*(i+1)+1];
        }
        i++;
    }
    return true;
}

} // namespace mariana
