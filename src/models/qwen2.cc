/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : models/qwen2.cc
 * Authors    : lqwang@pandora
 * Create Time: 2025-02-15:16:45:58
 * Description:
 * 
 */


#include <core/graph.h>
#include <core/node.h>
#include <core/function.h>

#include <models/qwen2.h>

#include <mariana_llm/mariana_llm.h>

#include <utils/sys.h>
#include <utils/json_utils.h>
#include <utils/mariana_define.h>
#include <utils/rapidjson/document.h>

#include <token/sentencepiece.h>

#include <absl/strings/match.h>
#include <absl/strings/str_format.h>

namespace mariana {

tensor_list Qwen2::_compute(ExeContext& context, const std::vector<int>& tokens, int32_t cache_len) {
    TRACE();
    AIResult result;
    Tensor position_ids = _get_position_ids(tokens, cache_len);
    Tensor input_ids({1, static_cast<int32_t>(tokens.size())}, DataOn::CPU, const_cast<int*>(tokens.data()), TypeMeta::make<int32_t>());
    Tensor attn_mask = _get_attn_mask(tokens, cache_len);
    KeyTensorMap key_tensor_map;
    key_tensor_map = {
        {"model.embed_tokens", {input_ids}},
        {"model.position_ids", {position_ids}},
        {"model.attn_mask", {attn_mask}},
    };
    tensor_list otensors = m_graph->forward(key_tensor_map, context);
    return otensors;
}

Tensor Qwen2::_get_position_ids(const std::vector<int>& tokens, int32_t cache_len) {
    Tensor postion_ids({1, static_cast<int32_t>(tokens.size())});
    int32_t* pi_ptr = postion_ids.mutable_ptr<int32_t>();
    for (uint32_t i = cache_len; i < postion_ids.total_size()+cache_len; ++i) {
        *pi_ptr = static_cast<int32_t>(i);
        pi_ptr += 1;
    }
    return postion_ids;
}

Tensor Qwen2::_get_attn_mask(const std::vector<int>& tokens, int32_t cache_len) {
    Tensor attn_mask({1, _num_atten_heads, static_cast<int32_t>(tokens.size()), static_cast<int32_t>(tokens.size())+cache_len});
    for (int n = 0; n < attn_mask.dim_at(0); ++n) {
        for (int ah = 0; ah < attn_mask.dim_at(1); ++ah) {
            for (int h = 0; h < attn_mask.dim_at(2); ++h) {
                for (int w = 0; w < attn_mask.dim_at(3); ++w) {
                    int32_t idx = n*attn_mask.stride_at(0)+ah*attn_mask.stride_at(1)
                                  + h*attn_mask.stride_at(2) + w*attn_mask.stride_at(3);
                    if (h < w) {
                        attn_mask.mutable_ptr<uint8_t>()[idx] = 0;
                    } else {
                        attn_mask.mutable_ptr<uint8_t>()[idx] = 1;
                    }
                }
            }
        }
    }
    return attn_mask;
}

bool Qwen2::load_token(const char* dir_path) {
    TRACE();
    AnyMap     token_param;
    std::string token_cfg_path = os_path_join(dir_path, "tokenizer_config.json");
    load_config(token_cfg_path.c_str(), token_param);
    
    std::string tokenizer_class;
    TRY_ANY_CAST(tokenizer_class, token_param.at("tokenizer_class"), return false);
    m_tokenizer = std::make_shared<SentencepieceTokenizer>();
    bool ok = m_tokenizer->load(dir_path, token_param);
    return ok;
}

bool Qwen2::make_graph(const char* dir_path, GptParams& gpt_params, ExeContext& context) {
    TRACE();
    this->load_token(dir_path);
    AnyMap     qwen2_param;
    std::string qwen2_param_config = os_path_join(dir_path, "config.json");
    bool ok = load_config(qwen2_param_config.c_str(), qwen2_param);
    ModelParam model_param;
    TRY_ANY_CAST(model_param.n_vocab, qwen2_param.at("vocab_size"), return false);
    TRY_ANY_CAST(model_param.n_layer, qwen2_param.at("num_hidden_layers"), return false);
    TRY_ANY_CAST(model_param.n_head, qwen2_param.at("num_attention_heads"), return false);
    _num_atten_heads = model_param.n_head;
    TRY_ANY_CAST(model_param.n_embd, qwen2_param.at("hidden_size"), return false);
    TRY_ANY_CAST(model_param.max_position_embeddings, qwen2_param.at("max_position_embeddings"), return false);
    TRY_ANY_CAST(model_param.tie_word_embeddings, qwen2_param.at("tie_word_embeddings"), return false);
    TRY_ANY_CAST(model_param.layer_norm_eps, qwen2_param.at("rms_norm_eps"), return false);
    TRY_ANY_CAST(model_param.num_key_value_heads, qwen2_param.at("num_key_value_heads"), return false);
    TRY_ANY_CAST(model_param.intermediate_size, qwen2_param.at("intermediate_size"), return false);
    TRY_ANY_CAST(model_param.hidden_act, qwen2_param.at("hidden_act"), return false);
    std::string safe_tensors = os_path_join(dir_path, "model.safetensors");
    SafeTensorsCallback callback = [](ModelParam::SafeTensorInfo&sti, ModelParam&param,
                                      const std::string&key)->void {
        if (absl::StartsWith(key, "model") || absl::StartsWith(key, "lm_head")) {
            param.sti_map[key] = sti;
            MVLOG(4)<<"read the qwen weight:"<<key;
        }
    };
    ok = ok &  _load_safetensors(safe_tensors.c_str(), model_param, callback);
    m_graph = std::make_shared<Graph>(gpt_params.n_threads);
    NodeSharedPtr inputs_position_ids_pass = m_graph->make_root(model_param, "model.position_ids");
    NodeSharedPtr inputs_embedding_pass = m_graph->make_root(model_param, "model.embed_tokens");
    NodeSharedPtr att_mask_pass = m_graph->make_root(model_param, "model.attn_mask");
    NodeSharedPtr att_mask = m_graph->make_node(OpCategory::AttMask, model_param, {att_mask_pass}, "model.attn_mask");
    NodeSharedPtr inputs_embedding = m_graph->make_node(OpCategory::GetRows, model_param, {inputs_embedding_pass}, "model.embed_tokens");
    int32_t rope_theta = -1;
    bool is_int32 = true;
    TRY_ANY_CAST(rope_theta, qwen2_param.at("rope_theta"), is_int32=false);
    if (is_int32 == true) {
        model_param.rope_theta = static_cast<float>(rope_theta);
    } else {
        TRY_ANY_CAST(model_param.rope_theta, qwen2_param.at("rope_theta"), pass);
    }
    if (qwen2_param.count("rope_scaling") != 0) {
        AnyMap rope_scaling;
        TRY_ANY_CAST(rope_scaling, qwen2_param.at("rope_scaling"), pass);
        if (rope_scaling.count("rope_type") == 0) {
            TRY_ANY_CAST(model_param.rope_type, rope_scaling.at("type"), pass);
        } else {
            TRY_ANY_CAST(model_param.rope_type, rope_scaling.at("rope_type"), pass);
        }
    }
    if (qwen2_param.count("partial_rotary_factor") != 0) {
        TRY_ANY_CAST(model_param.partial_rotary_factor, qwen2_param.at("partial_rotary_factor"), pass);
    }
    NodeSharedPtr rope_node = m_graph->make_node(OpCategory::ROPE, model_param, {inputs_position_ids_pass});
    model_param.q_weight_prefix = "q_proj";
    model_param.k_weight_prefix = "k_proj";
    model_param.v_weight_prefix = "v_proj";
    model_param.o_weight_prefix = "o_proj";
    NodeSharedPtr residual = inputs_embedding;
    for (int32_t dcl_idx = 0; dcl_idx < model_param.n_layer; ++dcl_idx) {
        // 1. Attention
        std::string name = absl::StrFormat("model.layers.%d.input_layernorm", dcl_idx);
        NodeSharedPtr input_ln_node = m_graph->make_node(OpCategory::RMSNorm, model_param, {residual}, name);
        name = absl::StrFormat("model.layers.%d.self_attn", dcl_idx);
        NodeSharedPtr attn_node = m_graph->make_node(OpCategory::SelfAtt, model_param, {input_ln_node, rope_node, att_mask}, name);
        NodeSharedPtr add_node = m_graph->make_node(OpCategory::Add, model_param, {residual, attn_node});
        name = absl::StrFormat("model.layers.%d.post_attention_layernorm", dcl_idx);
        NodeSharedPtr post_ln_node = m_graph->make_node(OpCategory::RMSNorm, model_param, {add_node}, name);
        // 2. MLP
        name = absl::StrFormat("model.layers.%d.mlp.gate_proj", dcl_idx);
        std::string hidden_act;
        TRY_ANY_CAST(hidden_act, qwen2_param.at("hidden_act"), pass);
        if (hidden_act == "silu") {
            model_param.act_cate = OpCategory::SiLU;
        } else {
            MLOG(FATAL)<<"Unknown hidden_act";   
        }
        NodeSharedPtr gate_proj = m_graph->make_node(OpCategory::MatMul, model_param, {post_ln_node}, name);
        model_param.act_cate = OpCategory::None;
        name = absl::StrFormat("model.layers.%d.mlp.up_proj", dcl_idx);
        NodeSharedPtr up_proj = m_graph->make_node(OpCategory::MatMul, model_param, {post_ln_node}, name);
        NodeSharedPtr mul_node = m_graph->make_node(OpCategory::Mul, model_param, {gate_proj, up_proj});
        name = absl::StrFormat("model.layers.%d.mlp.down_proj", dcl_idx);
        NodeSharedPtr down_proj = m_graph->make_node(OpCategory::MatMul, model_param, {mul_node}, name);
        residual = m_graph->make_node(OpCategory::Add, model_param, {add_node, down_proj});
    }
    NodeSharedPtr norm_node = m_graph->make_node(OpCategory::RMSNorm, model_param, {residual}, "model.norm");
    NodeSharedPtr lm_head = m_graph->make_leaf(OpCategory::MatMul, model_param, {norm_node}, "lm_head");
    model_param.release();
    return ok;
}

} // namespace mariana

