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

std::vector<int> Tiktoken::encode(const std::string& str) {
    return {};
}

std::string Tiktoken::decode(int id) {
    std::string word;
    TRY_STL(word = decoder_.at(id), return "");
    return word;
}

} // namespace mariana
