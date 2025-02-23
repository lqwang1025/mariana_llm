/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : llama_token_fast.h
 * Authors    : lqwang@pandora
 * Create Time: 2025-02-19:20:09:40
 * Description:
 *
 */

#ifndef __LLAMA_TOKEN_FAST_H__
#define __LLAMA_TOKEN_FAST_H__

#include <token/tokenizer.h>

namespace mariana {

class LlamaFastTokenizer : public Tokenizer {
public:
    LlamaFastTokenizer() = default;
    ~LlamaFastTokenizer() = default;
    virtual bool load(const std::string& filename, const AnyMap& param) override;
    virtual std::vector<int> encode(const std::string& str) override;
    virtual std::string decode(int id) override;
};

} // namespace mariana

#endif /* __LLAMA_TOKEN_FAST_H__ */

