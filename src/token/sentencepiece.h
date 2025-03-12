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

#include <utility>
#include <vector>
#include <unordered_map>
#include <map>

#include <token/tokenizer.h>

namespace mariana {

class SentencepieceTokenizer : public Tokenizer {
public:
    SentencepieceTokenizer() = default;
    ~SentencepieceTokenizer() = default;
    virtual bool load(const std::string& filename, const AnyMap& param) override;
    virtual void encode(const std::string& str, std::vector<int>& tokens) override;
    virtual std::string decode(int id) override;
private:
    std::map<std::pair<std::string, std::string>, int> _bpe_ranks;
    std::vector<std::string> _decoder;
};

} // namespace mariana

#endif /* __SENTENCEPIECE_H__ */

