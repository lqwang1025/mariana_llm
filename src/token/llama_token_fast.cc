/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : llama_token_fast.cc
 * Authors    : lqwang@pandora
 * Create Time: 2025-02-19:20:14:58
 * Description:
 * 
 */

#include <token/llama_token_fast.h>

#include <utils/sys.h>
#include <utils/mariana_define.h>

namespace mariana {

bool LlamaFastTokenizer::load(const std::string& filename, const AnyMap& param) {
    bool add_bos_token = false;
    if (param.count("add_bos_token")) {
        TRY_ANY_CAST(add_bos_token, param.at("add_bos_token"), pass);
    }
    bool add_eos_token = false;
    if (param.count("add_eos_token")) {
        TRY_ANY_CAST(add_eos_token, param.at("add_eos_token"), pass);
    }
    
    std::string eos_token = "";
    if (param.count("eos_token") && add_eos_token) {
        try {
            eos_token = ::absl::any_cast<std::string>(param.at("eos_token"));
        } catch(const absl::bad_any_cast &e) {
            AnyMap any_map = ::absl::any_cast<AnyMap>(param.at("eos_token"));
            eos_token = ::absl::any_cast<std::string>(any_map.at("content"));
        }
    }
    
    std::string bos_token = "";
    if (param.count("bos_token") && add_bos_token) {
        try {
            bos_token = ::absl::any_cast<std::string>(param.at("bos_token"));
        } catch(const absl::bad_any_cast &e) {
            AnyMap any_map = ::absl::any_cast<AnyMap>(param.at("bos_token"));
            bos_token = ::absl::any_cast<std::string>(any_map.at("content"));
        }
    }
    
    std::string chat_template = "";
    TRY_ANY_CAST(chat_template, param.at("chat_template"), pass);
    if (chat_template.empty() == false) {
        _chat_tmpl = new minja::chat_template(chat_template, bos_token, eos_token);
        minja::chat_template_inputs inputs;
        inputs.messages = json::parse(R"([
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"}
    ])");
        inputs.add_generation_prompt = true;
    //     inputs.tools = json::parse(R"([
    //     {"type": "function", "function": {"name": "google_search", "arguments": {"query": "2+2"}}}
    // ])");
    }
    
    std::string token_cfg_path = os_path_join(dir_path, "tokenizer_config.json");
    if file_exist("");
}

std::vector<int> LlamaFastTokenizer::encode(const std::string& str) {
    
}

std::string LlamaFastTokenizer::decode(int id) {
    
}

} // namespace mariana
