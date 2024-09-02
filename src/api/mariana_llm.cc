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

void* mariana_create_lmodel_handle(LModelCategory lmodel, ExeContext& context) {
    auto func_make = mariana::LModelHolder::search(lmodel);
    LModel* model = func_make();
    model->init(context.config_dir.c_str(), context);
    return model;
}

AIResult mariana_compute_lmodel_handle(void* handle, ExeContext& context) {
    LModel* model = static_cast<LModel*>(handle);
    return model->compute(context);
}

void mariana_destroy_lmodel_handle(void* handle, ExeContext& context) {
    LModel* model = static_cast<LModel*>(handle);
    delete model;
}

void mariana_llm_finit() {
    
}

} // namespace mariana
