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

#include <set>
#include <queue>
#include <string>
#include <cfloat>

#include <token/sentencepiece.h>
#include <token/unicode.h>

#include <utils/sys.h>
#include <utils/json_utils.h>
#include <utils/mariana_define.h>

#include <absl/strings/strip.h>
#include <absl/strings/str_split.h>

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
    if (param.count("eos_token")) {
        try {
            eos_token = ::absl::any_cast<std::string>(param.at("eos_token"));
        } catch(const absl::bad_any_cast &e) {
            AnyMap any_map = ::absl::any_cast<AnyMap>(param.at("eos_token"));
            eos_token = ::absl::any_cast<std::string>(any_map.at("content"));
        }
    }
    
    std::string bos_token = "";
    if (param.count("bos_token")) {
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
        std::string _eos_token = "";
        if (add_eos_token) {
            _eos_token = eos_token;
        }
        std::string _bos_token = "";
        if (add_bos_token) {
            _bos_token = bos_token;
        }
        _chat_tmpl = new minja::chat_template(chat_template, _bos_token, _eos_token);
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
        _special_pieces.insert({content, id});
        if (eos_token == content) {
            _stop_tokens.push_back(id);
        }
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
    std::vector<std::string> merges;
    TRY_ANY_CAST(merges, model.at("merges"), pass);
    for (size_t i = 0; i < merges.size(); ++i) {
        absl::string_view _tmp = merges[i];
        absl::ConsumePrefix(&_tmp, " ");
        absl::ConsumeSuffix(&_tmp, " ");
        std::vector<std::string> right_and_left = absl::StrSplit(_tmp, ' ');
        auto pair = std::make_pair(right_and_left[0], right_and_left[1]);
        _bpe_ranks.insert(std::make_pair(pair, i));
    }
    
    _decoder.resize(_pieces.size()+_special_pieces.size());
    for (auto& pieces : _pieces) {
        _decoder[pieces.second] = pieces.first;
    }
    for (auto& pieces : _special_pieces) {
        _decoder[pieces.second] = pieces.first;
    }
    
    // bytes_to_unicode
    std::unordered_map<uint8_t, wchar_t> b2u;
    auto _insert_range = [&](int start, int end) {
        for (int c = start; c <= end; c++) {
            b2u.insert({uint8_t(c), wchar_t(c)});
        }
    };

    b2u.clear();
    _insert_range(L'!', L'~');
    _insert_range(L'¡', L'¬');
    _insert_range(L'®', L'ÿ');

    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (b2u.find(uint8_t(b)) == b2u.end()) {
            b2u.insert({uint8_t(b), wchar_t(256 + n)});
            n++;
        }
    }
    for (auto e : b2u) {
        _u2b.insert({e.second, e.first});
    }
    
    return true;
}

void SentencepieceTokenizer::encode(const std::string& str, std::vector<int>& tokens) {
    std::vector<std::string> words = unicode_regex_split(str, _regexes);
    struct LlmSymbol {
        using index = int;
        index prev;
        index next;
        const char* text;
        size_t n;
    };

    struct LlmBigramBPE {
        struct comparator {
            bool operator()(const LlmBigramBPE & l, const LlmBigramBPE & r) const {
                return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
            }
        };
        using queue_storage = std::vector<LlmBigramBPE>;
        using queue = std::priority_queue<LlmBigramBPE, queue_storage, comparator>;
        LlmSymbol::index left;
        LlmSymbol::index right;
        std::string text;
        int rank;
        size_t size;
    };

    LlmBigramBPE::queue work_queue;
    
    auto add_new_bigram = [&](int left, int right, const std::vector<LlmSymbol>& symbols) ->void {
        if (left == -1 || right == -1) {
            return;
        }
        std::string left_token  = std::string(symbols[left].text,  symbols[left].n);
        std::string right_token = std::string(symbols[right].text, symbols[right].n);

        auto it = _bpe_ranks.find(std::make_pair(left_token, right_token));
        if (it == _bpe_ranks.end()) {
            return;
        }
        int rank_found = it->second;
        LlmBigramBPE bigram;
        bigram.left  = left;
        bigram.right = right;
        bigram.text  = left_token + right_token;
        bigram.size  = left_token.size() + right_token.size();
        bigram.rank  = rank_found;
        work_queue.push(bigram);
    };
    int final_prev_index = -1;
    std::vector<LlmSymbol> symbols_final;
    std::vector<LlmSymbol> symbols;
    for (auto& word : words) {
        symbols.clear();
        size_t offset = 0;
        int index = 0;
        std::queue<std::string> word_queue;
        while (offset < word.size()) {
            LlmSymbol sym;
            size_t char_len = std::min(word.size()-offset, (size_t)unicode_len_utf8(word[offset]));
            sym.text = word.c_str() + offset;
            sym.n = char_len;
            offset += sym.n;
            sym.prev = index - 1;
            sym.next = offset == word.size() ? -1 : index + 1;
            index++;
            symbols.emplace_back(sym);
        }
        
        for (size_t i = 1; i < symbols.size(); ++i) {
            add_new_bigram(i - 1, i, symbols);
        }

        while (!work_queue.empty()) {
            auto bigram = work_queue.top();
            work_queue.pop();
            auto & left_symbol = symbols[bigram.left];
            auto & right_symbol = symbols[bigram.right];
            if (left_symbol.n == 0 || right_symbol.n == 0) {
                continue;
            }
            std::string left_token = std::string(left_symbol.text, left_symbol.n);
            std::string right_token = std::string(right_symbol.text, right_symbol.n);
            if (left_token + right_token != bigram.text) {
                continue;  // Skip this bigram if it's outdated
            }

            // merge the right sym into the left one
            left_symbol.n += right_symbol.n;
            right_symbol.n = 0;

            // remove the right sym from the chain
            left_symbol.next = right_symbol.next;
            if (right_symbol.next >= 0) {
                symbols[right_symbol.next].prev = bigram.left;
            }

            add_new_bigram(left_symbol.prev, bigram.left, symbols); // left side of current symbol
            add_new_bigram(bigram.left, left_symbol.next, symbols); // right side of current symbol
        }

        // add the finished tokens to the final list keeping correct order for next and prev
        for (auto & sym : symbols) {
            if (sym.n > 0) {
                sym.prev = final_prev_index;
                sym.next = -1;
                if (final_prev_index != -1) {
                    symbols_final[final_prev_index].next = symbols_final.size();
                }
                symbols_final.emplace_back(sym);
                final_prev_index = symbols_final.size() - 1;
            }
        }   
    }
    
    symbols = symbols_final;
    if (!symbols.empty()) {
        for (int i = 0; i != -1; i = symbols[i].next) {
            auto & symbol = symbols[i];
            if (symbol.n == 0) {
                continue;
            }
                
            const std::string str = std::string(symbol.text, symbol.n);
            const auto token = _pieces.at(str);
            tokens.push_back(token);
        }
    }
}

static std::wstring utf8_to_wstring(const std::string& str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> myconv;
    return myconv.from_bytes(str);
}


std::string SentencepieceTokenizer::decode(int id) {
    if (id >= static_cast<int>(_decoder.size())) {
        MLOG(INFO)<<"DD:"<<id;
        return "";
    }
    std::wstring w = utf8_to_wstring(_decoder.at(id));
    std::string r;
    for (wchar_t c : w) {
        if (_u2b.find(c) != _u2b.end()) {
            r.push_back(char(_u2b.at(c)));
        }
    }
    return r;
}

} // namespace mariana
