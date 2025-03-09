/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : sentencepiece.cc
 * Authors    : lqwang@SMT23090002
 * Create Time: 2025-03-03:13:34:53
 * Description:
 * 
 */

#include <string>
#include <locale>
#include <codecvt>

#include <token/sentencepiece.h>
#include <token/unicode.h>

#include <utils/sys.h>
#include <utils/json_utils.h>
#include <utils/mariana_define.h>

namespace mariana {

 
bool SentencepieceTokenizer::load(const std::string& filename, const AnyMap& param) {
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
    }
    std::string token_cfg_path = os_path_join(filename, "tokenizer.json");
    AnyMap token_param;
    load_config(token_cfg_path.c_str(), token_param);
    
    AnyMap pre_tokenizer;
    TRY_ANY_CAST(pre_tokenizer, token_param.at("pre_tokenizer"), pass);
    std::vector<AnyMap> pretokenizers;
    TRY_ANY_CAST(pretokenizers, pre_tokenizer.at("pretokenizers"), pass);
    _regexes.clear();
    for (auto& it : pretokenizers) {
        std::string type;
        TRY_ANY_CAST(type, it.at("type"), pass);
        if (type == "Split") {
            AnyMap pattern;
            TRY_ANY_CAST(pattern, it.at("pattern"), pass);
            std::string regex;
            TRY_ANY_CAST(regex, pattern.at("Regex"), pass);
            _regexes.push_back(regex);
        }
    }
    
    std::vector<AnyMap> added_tokens;
    TRY_ANY_CAST(added_tokens, token_param.at("added_tokens"), pass);
    for (auto& token : added_tokens) {
        std::string content;
        TRY_ANY_CAST(content, token.at("content"), pass);
        int32_t id;
        TRY_ANY_CAST(id, token.at("id"), pass);
        _pieces.insert({content, id});
    }
    AnyMap model;
    TRY_ANY_CAST(model, token_param.at("model"), pass);
    AnyMap vocabs;
    TRY_ANY_CAST(vocabs, model.at("vocab"), pass);
    for (auto& vocab : vocabs) {
        std::string content = vocab.first;
        int32_t idx;
        TRY_ANY_CAST(idx, vocab.second, pass);
        _pieces.insert({content, idx});
    }
    
    // std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    _decoder.resize(_pieces.size());
    for (auto& pieces : _pieces) {
        _decoder[pieces.second] = pieces.first;
        // std::wstring str = converter.from_bytes(pieces.first);
        // std::wstring ssstr = converter.to_bytes(str);
        // std::wcout<< pieces.second<<" "<<ssstr<<std::endl;
    }
    minja::chat_template_inputs inputs;
    inputs.messages = json::parse(R"([
        {"role": "user", "content": "Hello"}
    ])");
    inputs.add_generation_prompt = true;
    std::string prompt = _chat_tmpl->apply(inputs);
    std::string decoded;
    std::vector<uint32_t> cpts = unicode_cpts_from_utf8(prompt);
    for (const auto cpt : cpts) {
        const auto utf8 = unicode_byte_to_utf8(cpt);
        decoded += unicode_utf8_to_byte(utf8);
    }
    MLOG(INFO)<<decoded;
    
    // std::vector<std::string> tests = unicode_regex_split(prompt, _regexes);
    // for (auto it : tests)
    //     MLOG(INFO)<<it;
    return true;
}

std::vector<int> SentencepieceTokenizer::encode(const std::string& str) {
    
}

std::string SentencepieceTokenizer::decode(int id) {
    
}


} // namespace mariana
