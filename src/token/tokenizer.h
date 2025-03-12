/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : token/tokenizer.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-05-31:13:09:51
 * Description:
 *
 */

#ifndef __TOKENIZER_H__
#define __TOKENIZER_H__

#include <vector>
#include <string>
#include <unordered_map>

#include <unordered_map>
#include <absl/types/any.h>

#include <token/minjia/chat-template.hpp>

namespace mariana {

using AnyMap = std::unordered_map<std::string, ::absl::any>;

class Tokenizer {
public:
    Tokenizer() = default;
    virtual ~Tokenizer() {
        if (_chat_tmpl != nullptr) {
            delete _chat_tmpl;
            _chat_tmpl = nullptr;
        }
    }
    virtual bool load(const std::string& filename, const AnyMap& param) = 0;
    virtual std::vector<int> encode(const std::string& str);
    virtual void encode(const std::string& str, std::vector<int>& tokens)=0;
    virtual std::string decode(int id) = 0;
    virtual std::string apply_chat_template(const std::string& str);
protected:    
    minja::chat_template *_chat_tmpl = nullptr;
    std::unordered_map<std::string, int> _special_pieces;
    std::unordered_map<std::string, int> _pieces;
    std::vector<std::string> _regexes;
};

class Tiktoken : public Tokenizer {
public:
    Tiktoken() = default;
    virtual ~Tiktoken() = default;
    virtual bool load(const std::string& filename, const AnyMap& param) override;
    virtual void encode(const std::string& str, std::vector<int>& tokens) override;
    virtual std::string decode(int id) override;
protected:
    std::unordered_map<std::string, int> encoder_;
    std::vector<std::string> decoder_;
};

} // namespace mariana 

#endif /* __TOKENIZER_H__ */

