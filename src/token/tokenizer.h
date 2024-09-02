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

namespace mariana {

class Tokenizer {
public:
    Tokenizer() = default;
    virtual ~Tokenizer() = default;
    virtual bool load(const std::string& filename) = 0;
    virtual std::vector<int> encode(const std::string& str) = 0;
    virtual std::string decode(int id) = 0;
};

class Tiktoken : public Tokenizer {
public:
    Tiktoken() = default;
    virtual ~Tiktoken() = default;
    virtual bool load(const std::string& filename) override;
    virtual std::vector<int> encode(const std::string& str) override;
    virtual std::string decode(int id) override;
protected:
    std::unordered_map<std::string, int> encoder_;
    std::vector<std::string> decoder_;
};

} // namespace mariana 

#endif /* __TOKENIZER_H__ */

