/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : api/mariana_llm.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-08-30:15:07:46
 * Description:
 * 
 */

#include <mariana_llm/mariana_llm.h>

#include <models/lmodels.h>
#include <models/init_lmodels_module.h>

namespace mariana {

void mariana_llm_init() {
    init_lmodels_module();
}

GptModel* mariana_create_lmodel(GptParams& gpt_params) {
    auto func_make = mariana::LModelHolder::search(gpt_params.lmodel);
    LModel* model = func_make();
    ExeContext* context = new ExeContext{};
    GptModel* gpt_model = new GptModel{};
    model->init(gpt_params.config_dir.c_str(), gpt_params, *context);
    gpt_model->handle = model;
    gpt_model->context = context;
    return gpt_model;
}

bool mariana_generate_lmodel(GptModel* gpt_model, AIResult& result) {
    LModel* model = static_cast<LModel*>(gpt_model->handle);
    bool ok = model->generate(*gpt_model->context, result);
    return ok;
}

void mariana_destroy_lmodel(GptModel* gpt_model) {
    LModel* model = static_cast<LModel*>(gpt_model->handle);
    delete model;
    delete gpt_model->context;
    delete gpt_model;
}

void mariana_llm_finit() {
    
}

} // namespace mariana
