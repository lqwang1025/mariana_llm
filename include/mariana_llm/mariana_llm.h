/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : mariana_llm/mariana_llm.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-05-28:15:27:45
 * Description:
 *
 */

#ifndef __MARIANA_LLM_H__
#define __MARIANA_LLM_H__

#include <string>
#include <vector>
#include <cstdint>

#include <mariana_llm/mariana_llm_impl.h>

namespace mariana {

enum class LModelCategory : int16_t {
    None = 0,
    Bert = 1,
    GroundingDINO = 2
};

struct PostProcessInfo {
    float box_threshold = 0.f;
    float text_threshold = 0.f;
};

struct ExeContext {
    std::string prompt{""};
    int32_t max_text_len{0};
    AIImage image;
    PostProcessInfo post_info;
};

struct GptParams {
    int32_t n_threads{-1};
    std::string config_dir{""};
    DataOn backend = DataOn::NONE;
    LModelCategory lmodel = LModelCategory::None;
};

struct GptModel {
    ExeContext* context = nullptr;
    void* handle = nullptr;
};

void mariana_llm_init();

GptModel* mariana_create_lmodel(GptParams& gpt_params);

AIResult mariana_compute_lmodel(GptModel* gpt_model);

void mariana_destroy_lmodel(GptModel* gpt_model);

void mariana_llm_finit();

} // namespace mariana

#endif /* __MARIANA_LLM_H__ */

