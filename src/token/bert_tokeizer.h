/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : token/bert_tokeizer.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-12:17:04:00
 * Description:
 *
 */

#ifndef __BERT_TOKEIZER_H__
#define __BERT_TOKEIZER_H__

#include <token/tokenizer.h>

namespace mariana {

class BertTokenizer : public Tiktoken {
public:
    BertTokenizer() = default;
    ~BertTokenizer() = default;
    virtual std::vector<int> encode(const std::string& str) override;
private:
    std::vector<int> word_piece(const std::string& token);
private:
    const std::string k_unk_token_ = "[UNK]";
    const std::string k_mask_token_ = "[MASK]";
    const std::string k_sep_token_ = "[SEP]";
    const std::string k_pad_token_ = "[PAD]";
    const std::string k_cls_token_ = "[CLS]";
};

} // namespace mariana

#endif /* __BERT_TOKEIZER_H__ */

