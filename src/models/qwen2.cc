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
#include <utils/mariana_define.h>
#include <utils/rapidjson/document.h>

#include <token/llama_token_fast.h>

#include <absl/strings/match.h>
#include <absl/strings/str_format.h>

namespace mariana {

AIResult Qwen2::compute(ExeContext& context) {
    
}

bool Qwen2::load_token(const char* dir_path) {
    AnyMap     token_param;
    std::string token_cfg_path = os_path_join(dir_path, "tokenizer_config.json");
    this->_load_config(token_cfg_path.c_str(), token_param);
    
    std::string tokenizer_class;
    TRY_ANY_CAST(tokenizer_class, token_param.at("tokenizer_class"), return false);
    m_tokenizer = std::make_shared<LlamaFastTokenizer>();
    bool ok = m_tokenizer->load(dir_path, token_param);
    return ok;
}

bool Qwen2::make_graph(const char* dir_path, GptParams& gpt_params, ExeContext& context) {
    TRACE();
    this->load_token(dir_path);
    AnyMap     qwen2_param;
    std::string qwen2_param_config = os_path_join(dir_path, "config.json");
    bool ok = _load_config(qwen2_param_config.c_str(), qwen2_param);
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
    NodeSharedPtr inputs_embedding_pass = m_graph->make_root(model_param, "model.embed_tokens");
    NodeSharedPtr inputs_embedding = m_graph->make_node(OpCategory::GetRows, model_param, {inputs_embedding_pass}, "model.text_backbone.embeddings.word_embeddings");
    MLOG(INFO)<<"DDDDDDDd:"<<model_param.hidden_act;
    return ok;
}

} // namespace mariana

