/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : mariana_llm/mariana_llm_impl.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-08-28:09:03:12
 * Description:
 *
 */

#ifndef __MARIANA_LLM_IMPL_H__
#define __MARIANA_LLM_IMPL_H__

#include <string>
#include <vector>
#include <cstdint>

namespace mariana {

struct Point2D {
    float x = 0.f;
    float y = 0.f;
};

struct Rect2D {
    Point2D tl;
    Point2D br;
    float height() const;
    float width() const;
    float area() const;
    Point2D cxy() const;
};

struct SResult {
    float score = 0.f;
    Rect2D bbox;
    std::vector<std::string> prompts;
    std::vector<int64_t> logits;
};

struct AIResult {
    std::vector<SResult> results;
    std::string id;
};

enum class DataOn : uint8_t {
    NONE = 0,
    CPU  = 1,
    GPU  = 2
};

struct AIImage { // default format is nhwc
    int32_t batch = 0;
    int32_t height = 0;
    int32_t width = 0;
    int32_t channel = 0;
    DataOn device = DataOn::NONE;
    uint8_t* data = nullptr;
};

} // namespace mariana

#endif /* __MARIANA_LLM_IMPL_H__ */

