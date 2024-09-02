/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : bert_tokeizer.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-12:17:05:26
 * Description:
 * 
 */

#include <token/bert_tokeizer.h>

namespace mariana {

std::vector<int> BertTokenizer::encode(const std::string& str) {
    std::vector<int> ids = {101};
    std::vector<std::string> tokens;
    std::string current_token;
    size_t i = 0;
    while (i < str.size()) {
        current_token.clear();
        unsigned char c = static_cast<unsigned char>(str[i]);
        
        if ((c & 0x80) != 0) { // handle multi-byte UTF-8 characters
            unsigned char mask = 0xE0; // 1110 0000 for 3-byte char
            if ((c & mask) == mask) {
                current_token = str.substr(i, 3);
                i += 3;
            } else {
                ++i;
                continue;
            }
        } else if (std::isalnum(c)) { // handle continuous sequence of letters and digits
            while (i < str.size() && std::isalnum(static_cast<unsigned char>(str[i]))) {
                current_token += std::tolower(str[i]);
                ++i;
            }
        } else if (std::ispunct(c)) { // handle punctuation and symbols
            current_token = str[i];
            ++i;
        } else if (std::isspace(c)) { // handle space, tab, enter
            ++i;
            continue;
        } else { // handle any other single-byte characters
            current_token = str[i];
            ++i;
        }
        if (!current_token.empty()) {
            tokens.push_back(current_token);
        }
    }
    for (auto& token : tokens) {
        for (auto& id : word_piece(token)) {
            ids.push_back(id);
        }
    }
    ids.push_back(102);
    return ids;
}

std::vector<int> BertTokenizer::word_piece(const std::string& token) {
    auto it = encoder_.find(token);
    if (it != encoder_.end()) {
        return {it->second};
    }
    std::vector<int> ids;
    std::string current = token;
    while (!current.empty()) {
        int match_id = -1;
        size_t match_pos = 0;
        for (int len = current.size(); len > 0; --len) {
            std::string candidate = current.substr(0, len);
            if (!ids.empty()) {
                candidate = "##" + candidate;
            }
            auto it = encoder_.find(candidate);
            if (it != encoder_.end()) {
                match_id = it->second;
                match_pos = len;
                break;
            }
        }
        if (match_id == -1) { // [UNK]
            ids.push_back(encoder_.at(k_unk_token_));
            break;
        }
        ids.push_back(match_id);
        current = current.substr(match_pos); // not first word, adding ## prefix
    }
    return ids;
}

} // namespace mariana
