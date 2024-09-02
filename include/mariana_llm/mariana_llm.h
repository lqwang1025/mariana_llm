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

struct RuntimeInfo {
    uint32_t feature_height = 0;
    uint32_t feature_width  = 0;
    void*    anything       = nullptr;
    std::vector<uint32_t> img_hws;
};

struct ExeContext {
    std::string prompt{""};
    int32_t max_text_len{0};
    int32_t n_threads{-1};
    std::string config_dir{""};
    AIImage image;
    RuntimeInfo runtime_info;
};

void mariana_llm_init();

void* mariana_create_lmodel_handle(LModelCategory lmodel, ExeContext& context);

AIResult mariana_compute_lmodel_handle(void* handle, ExeContext& context);

void mariana_destroy_lmodel_handle(void* handle, ExeContext& context);

void mariana_llm_finit();

} // namespace mariana

#endif /* __MARIANA_LLM_H__ */

