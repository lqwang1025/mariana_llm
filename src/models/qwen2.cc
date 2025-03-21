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
#include <core/tensor_utils.h>

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

AIResult Qwen2::compute(ExeContext& context) {
    TRACE();
    AIResult result;
    std::string str = R"([{"role": "system", "content": "你是一个有用的助手。"},
                {"role": "user", "content": "给我介绍一下大型语言模型 transformers。"}
               ])";
    std::string prompt = m_tokenizer->apply_chat_template(str);
    std::vector<int> tokens = m_tokenizer->encode(prompt);
    Tensor position_ids = _get_position_ids(tokens);
    Tensor input_ids({1, static_cast<int32_t>(tokens.size())}, DataOn::CPU, tokens.data(), TypeMeta::make<int32_t>());
    KeyTensorMap key_tensor_map;
    key_tensor_map = {
        {"model.embed_tokens", {input_ids}},
        {"model.position_ids", {position_ids}},
    };
    
    tensor_list otensors = m_graph->forward(key_tensor_map, context);
    return result;
}

Tensor Qwen2::_get_position_ids(const std::vector<int>& tokens) {
    Tensor postion_ids({1, static_cast<int32_t>(tokens.size())});
    for (uint32_t i = 0; i < postion_ids.total_size(); ++i) {
        postion_ids.mutable_ptr<int32_t>()[i] = static_cast<int32_t>(i);
    }    
    return postion_ids;
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
        if (absl::StartsWith(key, "model")) {
            param.sti_map[key] = sti;
            MVLOG(4)<<"read the qwen weight:"<<key;
        }
    };
    ok = ok &  _load_safetensors(safe_tensors.c_str(), model_param, callback);
    m_graph = std::make_shared<Graph>(gpt_params.n_threads);
    NodeSharedPtr inputs_position_ids_pass = m_graph->make_root(model_param, "model.position_ids");
    NodeSharedPtr inputs_embedding_pass = m_graph->make_root(model_param, "model.embed_tokens");
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
    for (int32_t dcl_idx = 0; dcl_idx < 1// model_param.n_layer
             ; ++dcl_idx) {
        std::string name = absl::StrFormat("model.layers.%d.input_layernorm", dcl_idx);
        NodeSharedPtr ln_node = m_graph->make_node(OpCategory::RMSNorm, model_param, {inputs_embedding}, name);
    }
    return ok;
}

} // namespace mariana

