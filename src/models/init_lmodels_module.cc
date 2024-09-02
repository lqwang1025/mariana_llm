/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : models/init_lmodels_module.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-18:11:05:52
 * Description:
 * 
 */

#include <models/all.h>
#include <models/lmodels_type.h>
#include <models/init_lmodels_module.h>
#include <core/init_core_module.h>
#include <ops/init_ops_module.h>

namespace mariana {

#define ADD_LMODEL(identity, type)                                      \
    static auto __##identity##_make = []()->LModel* { return new type{}; }; \
    LModelHolder::add_LModel(LModelCategory::identity, __##identity##_make)

static void _register_models() {
    ADD_LMODEL(GroundingDINO, GroundingDINO);
}

void init_lmodels_module() {
    init_core_module();
    init_ops_module();
    _register_models();
}

void uninit_lmodels_module() {
    
}

} // namespace mariana
