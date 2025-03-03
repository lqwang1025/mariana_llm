/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : token/sentencepiece.h
 * Authors    : lqwang@SMT23090002
 * Create Time: 2025-03-03:13:26:33
 * Description:
 *
 */

#ifndef __SENTENCEPIECE_H__
#define __SENTENCEPIECE_H__

#include <vector>
#include <unordered_map>

#include <token/tokenizer.h>

namespace mariana {

class SentencepieceTokenizer : public Tokenizer {
public:
    SentencepieceTokenizer() = default;
    ~SentencepieceTokenizer() = default;
    virtual bool load(const std::string& filename, const AnyMap& param) override;
    virtual std::vector<int> encode(const std::string& str) override;
    virtual std::string decode(int id) override;
private:
    std::unordered_map<std::string, int> _pieces;
    std::vector<std::string> _decoder;
};

} // namespace mariana

#endif /* __SENTENCEPIECE_H__ */

