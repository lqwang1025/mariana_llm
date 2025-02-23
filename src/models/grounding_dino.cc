/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : models/grounding_dino.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-18:11:19:46
 * Description:
 * 
 */

#include <set>
#include <cmath>
#include <fstream>
#include <sstream>

#include <core/graph.h>
#include <core/node.h>
#include <core/function.h>
#include <core/tensor_utils.h>

#include <ops/ops.h>
#include <ops/backend/cpu/grounding_dino_utils.h>

#include <models/grounding_dino.h>
#include <token/bert_tokeizer.h>
#include <mariana_llm/mariana_llm.h>

#include <utils/sys.h>
#include <utils/mariana_define.h>
#include <utils/rapidjson/document.h>

#include <absl/strings/match.h>
#include <absl/strings/str_format.h>

namespace mariana {

AIResult GroundingDINO::compute(ExeContext& context) {
    TRACE();
    // TODO: batch inputs!!!!
    std::vector<int32_t> token = m_tokenizer->encode(context.prompt);
    std::vector<int32_t> token_type;
    std::vector<uint8_t> attention_mask;
    int32_t input_size = token.size();
    int32_t trunc_len =  input_size <= context.max_text_len ? input_size : context.max_text_len;
    token.resize(trunc_len);
    token_type.resize(trunc_len, 0);
    attention_mask.resize(trunc_len);

    for (int32_t i = 0; i < input_size; ++i) {
        attention_mask[i] = token[i]>0 ? 1 : 0;
    }
    std::vector<int32_t> position_ids;
    std::vector<uint8_t> text_satt_mask_v; // kxk martrix
    _generate_masks_with_special_tokens_and_transfer_map(token, position_ids, text_satt_mask_v);
    
    Tensor pos_ids({1, trunc_len}, DataOn::CPU, position_ids.data(), TypeMeta::make<int32_t>());
    Tensor token_type_ids({1, trunc_len}, DataOn::CPU, token_type.data(), TypeMeta::make<int32_t>());
    
    Tensor input_ids({1, trunc_len}, DataOn::CPU, token.data(), TypeMeta::make<int32_t>());
    Tensor text_satt_mask({1, 1, trunc_len, trunc_len}, DataOn::CPU, text_satt_mask_v.data(), TypeMeta::make<uint8_t>());
    Tensor image({context.image.batch, context.image.height, context.image.width, context.image.channel},
                 context.image.device, context.image.data, TypeMeta::make<uint8_t>(), false);
    _pre_process(image, m_image_tensor);
    KeyTensorMap key_tensor_map;
    key_tensor_map = {
        {"model.text_backbone.embeddings.word_embeddings", {input_ids}},
        {"model.text_backbone.embeddings.position_embeddings", {pos_ids}},
        {"model.text_backbone.embeddings.token_type_embeddings", {token_type_ids}},
        {"model.att_mask", {text_satt_mask}},
        {"model.backbone.conv_encoder.model.embeddings.patch_embeddings.projection", {m_image_tensor}}
    };
    tensor_list otensors = m_graph->forward(key_tensor_map, context);
    AIResult result;
    if (otensors.empty()) {
        return result;
    }
    Tensor bbox = otensors[0];
    Tensor scores = otensors[1];
    Tensor probs = otensors[2];
    
    _post_process(bbox, scores, probs, token, result, context);
    return result;
}

void GroundingDINO::_pre_process(const Tensor& input, Tensor& out) {
    int32_t n            = input.dim_at(0);
    int32_t h            = input.dim_at(1);
    int32_t w            = input.dim_at(2);
    int32_t c            = input.dim_at(3);
    float   min_org_size = static_cast<float>(std::min(h, w));
    float   max_org_size = static_cast<float>(std::max(h, w));
    int32_t size         = m_shortest_edge;
    int32_t max_size     = m_longest_edge;
    if (max_org_size / min_org_size * size > max_size) {
        size = static_cast<int32_t>(std::round(max_size*min_org_size / max_org_size));
    }
    int32_t oh = 0, ow = 0;
    if ((h <= w && h == size) || (w <= h && w == size)) {
        oh = h;
        ow = w;
    } else {
        if (w < h) {
            ow = size;
            float factor = static_cast<float>(h)/static_cast<float>(w);
            oh = static_cast<int32_t>(size*factor);
        } else {
            oh = size;
            float factor = static_cast<float>(w)/static_cast<float>(h);
            ow = static_cast<int32_t>(size*factor);
        }
    }
    int32_t pad_r = m_patch_size - ow % m_patch_size;
    int32_t pad_b = m_patch_size - oh % m_patch_size;
    out.try_realloc({n, c, oh+pad_b, ow+pad_r}, TypeMeta::make<float>());
    Function::_parallel_sync(m_graph->thread_pool(), out.total_size(), grounding_dino_pre_process,
                             std::ref(input), std::ref(m_means), std::ref(m_stds),
                             std::ref(m_rescale_factor), std::ref(pad_r), std::ref(pad_b), std::ref(out));
}

void GroundingDINO::_post_process(const Tensor& bbox, const Tensor& scores, const Tensor& probs, const std::vector<int32_t>& token, AIResult& result, ExeContext& context) {
    int32_t B = bbox.dim_at(0);
    int32_t N = bbox.dim_at(1);
    if (B != 1) {
        MLOG(ERROR)<<"Support 1 batch now!!";
        return;
    }
    const float box_threshold = context.post_info.box_threshold;
    const float text_threshold = context.post_info.text_threshold;
    for (int32_t b = 0; b < B; ++b) {
        for (int32_t n = 0; n < N; ++n) {
            float score = scores.data_at<float>(b*scores.stride_at(0)+n);
            if (box_threshold < score) {
                SResult res;
                res.score = score;
                res.bbox.tl.x = bbox.data_at<float>(b*bbox.stride_at(0)+n*bbox.stride_at(1)+0);
                res.bbox.tl.y = bbox.data_at<float>(b*bbox.stride_at(0)+n*bbox.stride_at(1)+1);
                res.bbox.br.x = bbox.data_at<float>(b*bbox.stride_at(0)+n*bbox.stride_at(1)+2);
                res.bbox.br.y = bbox.data_at<float>(b*bbox.stride_at(0)+n*bbox.stride_at(1)+3);
                for (int32_t l = 1; l < probs.dim_at(2); ++l) {
                    float prob = probs.data_at<float>(b*probs.stride_at(0)+n*probs.stride_at(1)+l);
                    if (text_threshold < prob) {
                        res.logits.push_back(token.at(l));
                        res.prompts.push_back(m_tokenizer->decode(token.at(l)));
                    }
                }
                result.results.push_back(res);
            }
        }
    }
}

bool GroundingDINO::_generate_masks_with_special_tokens_and_transfer_map(const std::vector<int32_t>& input_ids,
                                                                         std::vector<int32_t>& position_ids,
                                                                         std::vector<uint8_t>& text_satt_mask) {
	const std::set<int32_t> SPECIAL_TOKENS = {101, 102, 1012, 1029}; // # these correspond to [CLS], [SEP], . and ?
    
    std::vector<std::int_fast8_t> special_tokens_mask;
    const size_t len_input_ids = input_ids.size();
    special_tokens_mask.resize(len_input_ids, 0);
    std::vector<size_t> ids;
    for (size_t i = 0; i < len_input_ids; ++i) {
        if (SPECIAL_TOKENS.count(input_ids[i])) {
            ids.push_back(i);
        }
    }
    
    text_satt_mask.resize(len_input_ids*len_input_ids);
    position_ids.resize(len_input_ids, 0);
    for (size_t i = 0; i < len_input_ids; ++i) {
        for (size_t j = 0; j < len_input_ids; ++j) {
            text_satt_mask[i*len_input_ids+j] = (i==j) ? 1 : 0; // Diagonal matrix
        }
    }
    
    size_t previous_col = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
        const size_t& col = ids[i];
        if (col == 0 || col == len_input_ids-1) {
            text_satt_mask[col*len_input_ids+col] = true;
            position_ids[col] = 0;
        } else {
            for (size_t j = previous_col + 1; j <= col; ++j) {
                for (size_t k = previous_col + 1; k <= col; ++k) {
					text_satt_mask[j*len_input_ids + k] = true;
				}
				position_ids[j] = j - previous_col - 1;
            }
        }
        previous_col = col;
    }
    
    return true;
}

bool GroundingDINO::make_graph(const char* dir_path, GptParams& gpt_params, ExeContext& context) {
    TRACE();
    AnyMap     bert_param;
    AnyMap     dino_param;
    AnyMap     swin_param;
    ModelParam bert_model_param;
    ModelParam dino_model_param;
    ModelParam swin_model_param;
    bool ok = load_param(dir_path, bert_param, dino_param, swin_param,
                         bert_model_param, dino_model_param, swin_model_param);
    AnyMap     preprocessor_param;
    std::string preprocess_config = os_path_join(dir_path, "preprocessor_config.json");
    ok = ok && _load_config(preprocess_config.c_str(), preprocessor_param);
    if (!ok) {
        MLOG(ERROR)<<"Load config failed";
        return ok;
    }
    
    TRY_ANY_CAST(m_do_normalize, preprocessor_param.at("do_normalize"), return false);
    TRY_ANY_CAST(m_do_pad, preprocessor_param.at("do_pad"), return false);
    TRY_ANY_CAST(m_do_rescale, preprocessor_param.at("do_rescale"), return false);
    TRY_ANY_CAST(m_do_resize, preprocessor_param.at("do_resize"), return false);
    TRY_ANY_CAST(m_rescale_factor, preprocessor_param.at("rescale_factor"), return false);
    TRY_ANY_CAST(m_means, preprocessor_param.at("image_mean"), return false);
    TRY_ANY_CAST(m_stds, preprocessor_param.at("image_std"), return false);
    TRY_ANY_CAST(m_patch_size, swin_param.at("patch_size"), return false);
    
    AnyMap size_param;
    TRY_ANY_CAST(size_param, preprocessor_param.at("size"), return false);
    TRY_ANY_CAST(m_longest_edge, size_param.at("longest_edge"), return false);
    TRY_ANY_CAST(m_shortest_edge, size_param.at("shortest_edge"), return false);
    // 0. bert embedding
    TRY_ANY_CAST(bert_model_param.n_vocab, bert_param.at("vocab_size"), return false);
    TRY_ANY_CAST(bert_model_param.n_embd, bert_param.at("hidden_size"), return false);
    TRY_ANY_CAST(bert_model_param.pad_token_id, bert_param.at("pad_token_id"), return false);
    TRY_ANY_CAST(bert_model_param.max_position_embeddings, bert_param.at("max_position_embeddings"), return false);
    TRY_ANY_CAST(bert_model_param.layer_norm_eps, bert_param.at("layer_norm_eps"), return false);
    if (context.max_text_len == 0) {
        TRY_ANY_CAST(context.max_text_len, dino_param.at("max_text_len"), return false);
    }
    m_graph = std::make_shared<Graph>(gpt_params.n_threads);

    NodeSharedPtr inputs_embedding_pass = m_graph->make_root(bert_model_param, "model.text_backbone.embeddings.word_embeddings");
    NodeSharedPtr inputs_embedding = m_graph->make_node(OpCategory::GetRows, bert_model_param, {inputs_embedding_pass}, "model.text_backbone.embeddings.word_embeddings");
    
    NodeSharedPtr position_embedding_pass = m_graph->make_root(bert_model_param, "model.text_backbone.embeddings.position_embeddings");
    NodeSharedPtr position_embedding = m_graph->make_node(OpCategory::GetRows, bert_model_param, {position_embedding_pass}, "model.text_backbone.embeddings.position_embeddings");
    
    NodeSharedPtr token_type_embedding_pass = m_graph->make_root(bert_model_param, "model.text_backbone.embeddings.token_type_embeddings");
    NodeSharedPtr token_type_embedding = m_graph->make_node(OpCategory::GetRows, bert_model_param, {token_type_embedding_pass}, "model.text_backbone.embeddings.token_type_embeddings");
    
    NodeSharedPtr att_mask_pass = m_graph->make_root(bert_model_param, "model.att_mask");
    NodeSharedPtr att_mask = m_graph->make_node(OpCategory::AttMask, bert_model_param, {att_mask_pass}, "model.att_mask");
    
    NodeSharedPtr next = m_graph->make_node(OpCategory::Add, bert_model_param, {inputs_embedding, token_type_embedding});

    std::string pet;
    TRY_ANY_CAST(pet, bert_param.at("position_embedding_type"), ;);
    if (pet == "absolute") {
        next = m_graph->make_node(OpCategory::Add, bert_model_param, {next, position_embedding});
    }
    
    TRY_ANY_CAST(bert_model_param.n_layer, bert_param.at("num_hidden_layers"), return false);
    TRY_ANY_CAST(bert_model_param.n_head, bert_param.at("num_attention_heads"), return false);
    next = m_graph->make_node(OpCategory::LayerNorm, bert_model_param, {next}, "model.text_backbone.embeddings.LayerNorm");
    
    // 1. text_backbone bert
    for (int32_t i = 0; i < bert_model_param.n_layer; ++i) {
        NodeSharedPtr route = next;
        std::string name = absl::StrFormat("model.text_backbone.encoder.layer.%d.attention.self", i);    
        next = m_graph->make_node(OpCategory::SelfAtt, bert_model_param, {next, att_mask}, name);
        
        name = absl::StrFormat("model.text_backbone.encoder.layer.%d.attention.output.dense", i);
        next = m_graph->make_node(OpCategory::MatMul, bert_model_param, {next}, name);
        next = m_graph->make_node(OpCategory::Add, bert_model_param, {next, route});
        
        name = absl::StrFormat("model.text_backbone.encoder.layer.%d.attention.output.LayerNorm", i);
        NodeSharedPtr layer_norm2 = m_graph->make_node(OpCategory::LayerNorm, bert_model_param, {next}, name);
        
        name = absl::StrFormat("model.text_backbone.encoder.layer.%d.intermediate.dense", i);
        bert_model_param.act_cate = OpCategory::GELU;
        next = m_graph->make_node(OpCategory::MatMul, bert_model_param, {layer_norm2}, name);
        
        //next = m_graph->make_node(OpCategory::GELU, bert_model_param, {next}, name);
        
        name = absl::StrFormat("model.text_backbone.encoder.layer.%d.output.dense", i);
        bert_model_param.act_cate = OpCategory::None;
        next = m_graph->make_node(OpCategory::MatMul, bert_model_param, {next}, name);
        
        next = m_graph->make_node(OpCategory::Add, bert_model_param, {next, layer_norm2});
        
        name = absl::StrFormat("model.text_backbone.encoder.layer.%d.output.LayerNorm", i);
        next = m_graph->make_node(OpCategory::LayerNorm, bert_model_param, {next}, name);
    }

    // 2. text_projection
    NodeSharedPtr text_features = m_graph->make_node(OpCategory::MatMul, dino_model_param, {next}, "model.text_projection");

    // 3. swin-backbone
    int32_t swin_n_embd = 0;
    TRY_ANY_CAST(swin_n_embd, swin_param.at("embed_dim"), return false);
    TRY_ANY_CAST(swin_model_param.o_channels, swin_param.at("embed_dim"), return false);
    TRY_ANY_CAST(swin_model_param.i_channels, swin_param.at("num_channels"), return false);
    TRY_ANY_CAST(swin_model_param.image_size, swin_param.at("image_size"), return false);
    TRY_ANY_CAST(swin_model_param.patch_size, swin_param.at("patch_size"), return false);
    swin_model_param.strides[0]        = swin_model_param.patch_size;
    swin_model_param.strides[1]        = swin_model_param.patch_size;
    swin_model_param.kernel_size[0]    = swin_model_param.patch_size;
    swin_model_param.kernel_size[1]    = swin_model_param.patch_size;
    swin_model_param.conv_output_trans = true;
    next = m_graph->make_root(swin_model_param, "model.backbone.conv_encoder.model.embeddings.patch_embeddings.projection");
    next = m_graph->make_node(OpCategory::Conv2D, swin_model_param, {next}, "model.backbone.conv_encoder.model.embeddings.patch_embeddings.projection");
    TRY_ANY_CAST(swin_model_param.layer_norm_eps, swin_param.at("layer_norm_eps"), return false);
    next = m_graph->make_node(OpCategory::LayerNorm, swin_model_param, {next}, "model.backbone.conv_encoder.model.embeddings.norm");

    TRY_ANY_CAST(swin_model_param.n_layer, swin_param.at("num_layers"), return false);
    std::vector<int32_t> depths;
    TRY_ANY_CAST(depths, swin_param.at("depths"), return false);
    std::vector<int32_t> num_heads;
    TRY_ANY_CAST(num_heads, swin_param.at("num_heads"), return false);
    TRY_ANY_CAST(swin_model_param.window_size, swin_param.at("window_size"), return false);
    std::vector<int32_t> out_indices_v;
    TRY_ANY_CAST(out_indices_v, swin_param.at("out_indices"), return false);
    std::set<int32_t> out_indices(out_indices_v.begin(), out_indices_v.end());

    std::vector<std::string> stage_names;
    TRY_ANY_CAST(stage_names, swin_param.at("stage_names"), return false);
    std::vector<NodeSharedPtr> connect_nodes;
    NodeSharedPtr _ln_info_node = next;
    for (size_t i_layer = 0; i_layer < depths.size(); ++i_layer) {
        swin_model_param.n_head = num_heads[i_layer];
        swin_model_param.n_embd = swin_n_embd * pow(2, i_layer);
        MVLOG(4)<<"swin stage:"<<i_layer+1<<" nembd:"
                <<swin_model_param.n_embd<<" nhead:"<<swin_model_param.n_head;
        for (int32_t i = 0; i < depths[i_layer]; ++i) {
            if (i%2 != 0) {
                swin_model_param.shift_size = swin_model_param.window_size/2;
            } else {
                swin_model_param.shift_size = 0;
            }
            std::string name = absl::StrFormat("model.backbone.conv_encoder.model.encoder.layers.%d.blocks.%d", i_layer, i);
            next = m_graph->make_node(OpCategory::SwinLayer, swin_model_param, {next}, name);
            next->push_info_shared_nodes({_ln_info_node});
            NodeSharedPtr route = next;
            name = absl::StrFormat("model.backbone.conv_encoder.model.encoder.layers.%d.blocks.%d.layernorm_after", i_layer, i);
            next = m_graph->make_node(OpCategory::LayerNorm, swin_model_param, {next}, name);
            swin_model_param.act_cate = OpCategory::GELU;
            name = absl::StrFormat("model.backbone.conv_encoder.model.encoder.layers.%d.blocks.%d.intermediate.dense", i_layer, i);
            next = m_graph->make_node(OpCategory::MatMul, swin_model_param, {next}, name);
            swin_model_param.act_cate = OpCategory::None;
            name = absl::StrFormat("model.backbone.conv_encoder.model.encoder.layers.%d.blocks.%d.output.dense", i_layer, i);
            next = m_graph->make_node(OpCategory::MatMul, swin_model_param, {next}, name);
            next = m_graph->make_node(OpCategory::Add, swin_model_param, {next, route});
        }
        auto out_index = out_indices.find(i_layer+1);
        if (out_index != out_indices.end()) { // output
            std::string stage_name = stage_names[*out_index];
            std::string name = absl::StrFormat("model.backbone.conv_encoder.model.hidden_states_norms.%s", stage_name);
            NodeSharedPtr swin_out = m_graph->make_node(OpCategory::SwinStageOutput, swin_model_param, {next}, name);
            swin_out->push_info_shared_nodes({_ln_info_node});
            connect_nodes.push_back(swin_out);
        }
        if (i_layer < depths.size()-1) {
            swin_model_param.patch_merge_step = 2;
            next = m_graph->make_node(OpCategory::SwinPatchMerging, swin_model_param, {next});
            next->push_info_shared_nodes({_ln_info_node});
            std::string name = absl::StrFormat("model.backbone.conv_encoder.model.encoder.layers.%d.downsample.norm", i_layer);
            next = m_graph->make_node(OpCategory::LayerNorm, swin_model_param, {next}, name);
            _ln_info_node = next;
            name = absl::StrFormat("model.backbone.conv_encoder.model.encoder.layers.%d.downsample.reduction", i_layer);
            next = m_graph->make_node(OpCategory::MatMul, swin_model_param, {next}, name);
        }
    }

    // 4. dino backbone
    TRY_ANY_CAST(dino_model_param.layer_norm_eps, dino_param.at("layer_norm_eps"), return false);
    size_t i = 0;
    std::vector<NodeSharedPtr> feature_maps;
    for (; i < connect_nodes.size(); ++i) {
        std::string name = absl::StrFormat("model.input_proj_vision.%d.0", i);
        next = m_graph->make_node(OpCategory::Conv2D, dino_model_param, {connect_nodes[i]}, name);
        name = absl::StrFormat("model.input_proj_vision.%d.1", i);
        dino_model_param.groups = 32;
        next = m_graph->make_node(OpCategory::GroupNorm, dino_model_param, {next}, name);
        feature_maps.push_back(next);
    }
    int32_t num_feature_levels = 0;
    TRY_ANY_CAST(num_feature_levels, dino_param.at("num_feature_levels"), return false);
    if (i < (uint32_t)num_feature_levels) {
        size_t f_size = feature_maps.size();
        for (; i < (uint32_t)num_feature_levels; ++i) {
            dino_model_param.strides[0]        = 2;
            dino_model_param.strides[1]        = 2;
            dino_model_param.kernel_size[0]    = 3;
            dino_model_param.kernel_size[1]    = 3;
            dino_model_param.padding[0]        = 1;
            dino_model_param.padding[1]        = 1;
            dino_model_param.padding[2]        = 1;
            dino_model_param.padding[3]        = 1;
            if (i == f_size) {
                std::string name = absl::StrFormat("model.input_proj_vision.%d.0", i);
                next = m_graph->make_node(OpCategory::Conv2D, dino_model_param, {connect_nodes[connect_nodes.size()-1]}, name);
                name = absl::StrFormat("model.input_proj_vision.%d.1", i);
                dino_model_param.groups = 32;
                next = m_graph->make_node(OpCategory::GroupNorm, dino_model_param, {next}, name);
            } else {
                std::string name = absl::StrFormat("model.input_proj_vision.%d.0", i);
                next = m_graph->make_node(OpCategory::Conv2D, dino_model_param, {feature_maps[f_size-1]}, name);
                name = absl::StrFormat("model.input_proj_vision.%d.1", i);
                dino_model_param.groups = 32;
                next = m_graph->make_node(OpCategory::GroupNorm, dino_model_param, {next}, name);
            }
            feature_maps.push_back(next);
        }
    }

    // 4.1 dino backbone postion embedding
    std::string position_embedding_type{}; // 1. sine 2. learned
    TRY_ANY_CAST(position_embedding_type, dino_param.at("position_embedding_type"), return false);
    if (position_embedding_type == "sine") {
        TRY_ANY_CAST(dino_model_param.positional_embedding_temperature, dino_param.at("positional_embedding_temperature"), return false);
        size_t feature_maps_size = feature_maps.size();
        for (size_t i = 0; i < feature_maps_size; ++i) {
            next = m_graph->make_node(OpCategory::GroundingDinoSinePositionEmbedding, dino_model_param, {feature_maps[i]});
            feature_maps.push_back(next);
        }
    } else {
        MLOG(ERROR)<<"Unsupport position_embedding_type in grounding dino: "<<position_embedding_type;
        return false;
    }
    // 4.2 dino encoder
    next = m_graph->make_node(OpCategory::GroundingDinoEncoderBefore, dino_model_param, feature_maps);
    NodeSharedPtr _gdeb_info_node = next;
    int32_t encoder_layers = 0;
    TRY_ANY_CAST(encoder_layers, dino_param.at("encoder_layers"), return false);
    TRY_ANY_CAST(dino_model_param.n_head, dino_param.at("encoder_attention_heads"), return false);
    dino_model_param.n_head /= 2;
    TRY_ANY_CAST(dino_model_param.n_embd, dino_param.at("encoder_ffn_dim"), return false);
    dino_model_param.n_embd /= 2;
    std::string activation_function;
    TRY_ANY_CAST(activation_function, dino_param.at("activation_function"), return false);
    if (activation_function != "relu") {
        MLOG(ERROR)<<"Unsupport act:"<<activation_function;
        return false;
    }
    
    TRY_ANY_CAST(dino_model_param.encoder_attention_heads, dino_param.at("encoder_attention_heads"), return false);
    TRY_ANY_CAST(dino_model_param.encoder_n_points, dino_param.at("encoder_n_points"), return false);
    TRY_ANY_CAST(dino_model_param.num_feature_levels, dino_param.at("num_feature_levels"), return false);
    TRY_ANY_CAST(dino_model_param.d_model, dino_param.at("d_model"), return false);
    std::string name = absl::StrFormat("model.encoder.layers.%d", 0);
    next = m_graph->make_node(OpCategory::GroundingDinoEncoderLayer, dino_model_param, {next, text_features, att_mask, position_embedding_pass}, name);
    next->push_info_shared_nodes({_gdeb_info_node});
    for (int32_t i = 1; i < encoder_layers; ++i) {
        name = absl::StrFormat("model.encoder.layers.%d", i);
        next = m_graph->make_node(OpCategory::GroundingDinoEncoderLayer, dino_model_param, {next}, name);
        next->push_info_shared_nodes({_gdeb_info_node});
    }
    // 4.3 dino decoder
    TRY_ANY_CAST(dino_model_param.num_queries, dino_param.at("num_queries"), return false);
    TRY_ANY_CAST(dino_model_param.decoder_layers, dino_param.at("decoder_layers"), return false);
    TRY_ANY_CAST(dino_model_param.n_head, dino_param.at("decoder_attention_heads"), return false);
    TRY_ANY_CAST(dino_model_param.n_embd, dino_param.at("d_model"), return false);
    next = m_graph->make_node(OpCategory::GroundingDinoDecoderBefore, dino_model_param, {next, token_type_embedding_pass}, name);
    next->push_info_shared_nodes({_gdeb_info_node});
    next = m_graph->make_leaf(OpCategory::GroundingDinoForDetection, dino_model_param, {next}, name);

    bert_model_param.release();
    dino_model_param.release();
    swin_model_param.release();
    
    return true;
}

bool GroundingDINO::load_token(const char* dir_path) {
    std::string vocab_txt = os_path_join(dir_path, "vocab.txt");
    if (false == file_exist(vocab_txt)) {
        MLOG(ERROR)<<"vocab.txt is not exist in:"<<dir_path;
        return false;
    }
    m_tokenizer = std::make_shared<BertTokenizer>();
    AnyMap _place_holder;
    bool ok = m_tokenizer->load(vocab_txt, _place_holder);
    MLOG_IF(ERROR, !ok)<<" tokenizer load failed";
    return ok;
}

bool GroundingDINO::load_param(const char* dir_path, AnyMap& bert_param, AnyMap& dino_param,
                               AnyMap& swin_param, ModelParam& bert_model_param,
                               ModelParam& dino_model_param, ModelParam& swin_model_param) {
    std::string safe_tensors = os_path_join(dir_path, "model.safetensors");
    SafeTensorsCallback bert_callback = [](ModelParam::SafeTensorInfo&sti, ModelParam&param,
                                           const std::string&key)->void {
        if (absl::StartsWith(key, "model.text_backbone")) {
            param.sti_map[key] = sti;
            MVLOG(4)<<"read the bert weight:"<<key;
        }
    };
    bool ok =  _load_safetensors(safe_tensors.c_str(), bert_model_param, bert_callback);

    SafeTensorsCallback swin_callback = [](ModelParam::SafeTensorInfo&sti, ModelParam&param,
                                           const std::string&key)->void {
        if (absl::StartsWith(key, "model.backbone")) {
            param.sti_map[key] = sti;
            MVLOG(4)<<"read the swin weight:"<<key;
        }
    };
    ok = ok & _load_safetensors(safe_tensors.c_str(), swin_model_param, swin_callback);
    
    SafeTensorsCallback dino_callback = [](ModelParam::SafeTensorInfo&sti, ModelParam&param,
                                           const std::string&key)->void {
        if (!absl::StartsWith(key, "model.text_backbone") &&
            !absl::StartsWith(key, "model.backbone")) {
            param.sti_map[key] = sti;
            MVLOG(4)<<"read the dino weight:"<<key;
        }
    };
    ok = ok & _load_safetensors(safe_tensors.c_str(), dino_model_param, dino_callback);
    
    std::string dino_config = os_path_join(dir_path, "config.json");
    ok = ok && _load_config(dino_config.c_str(), dino_param);
    
    std::string bert_config = os_path_join(dir_path, "bert/config.json");
    ok = ok && _load_config(bert_config.c_str(), bert_param);

    std::string swin_config = os_path_join(dir_path, "swin/config.json");
    ok = ok && _load_config(swin_config.c_str(), swin_param);
    
    ok = ok && load_token(dir_path);
    MLOG_IF(ERROR, !ok)<<"load param failed";
    return ok;
}

} // namespace mariana

