/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : tokenizer.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-01:06:09:50
 * Description:
 * 
 */

#include <fstream>
// #include <codecvt>
// #include <locale>

#include <token/tokenizer.h>
#include <utils/mariana_define.h>

namespace mariana {

std::string Tokenizer::apply_chat_template(const std::string& str) {
    if (_chat_tmpl == nullptr) {
        return str;
    }
    minja::chat_template_inputs inputs;
    inputs.messages = json::parse(str);
    inputs.add_generation_prompt = true;
    std::string prompt = _chat_tmpl->apply(inputs);
    return prompt;
}

std::vector<int> Tokenizer::encode(const std::string& prompt) {
    std::vector<int> tokens;
    size_t start = 0;
    if (false == _special_pieces.empty()) {
        for (size_t i = 0; i < prompt.length(); ++i) {
            for (auto it : _special_pieces) {
                auto special_token = it.first;
                if (i + special_token.length() <= prompt.length() && prompt.substr(i, special_token.length()) == special_token) {
                    if (i > start) {
                        encode(prompt.substr(start, i-start), tokens);
                    }
                    tokens.push_back(it.second);
                    start = i + special_token.length();
                    i = start - 1;
                    break;
                }
            }
        }
        if (start > prompt.length()) {
            encode(prompt.substr(start), tokens);
        }
    } else {
        encode(prompt, tokens);
    }
    return tokens;
}

bool Tiktoken::load(const std::string& filename, const AnyMap& param) {
    std::ifstream tok_file(filename);
    if (!tok_file.good()) {
        MLOG(ERROR)<<"Open file "<<filename<<" failed!";
        return false;
    }
    std::string token;
    // std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    // std::wstring wtoken = converter.from_bytes(token);
    while (tok_file >> token) {
        encoder_[token] = static_cast<int>(decoder_.size());
        decoder_.push_back(token);
    }
    tok_file.close();
    return true;
}

void Tiktoken::encode(const std::string& str, std::vector<int>& tokens) {
    return;
}

std::string Tiktoken::decode(int id) {
    std::string word;
    TRY_STL(word = decoder_.at(id), return "");
    return word;
}

} // namespace mariana
