/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : data_type.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-17:10:54:22
 * Description:
 * 
 */

#include <core/data_type.h>
#include <utils/mariana_define.h>

namespace mariana {

detail::TypeMetaData* TypeMeta::type_meta_datas() {
    static detail::TypeMetaData instances[UINT8_MAX+1] = {
        detail::TypeMetaData(TypeUninitIndex, 0, "Uninit"),
#define SCALAR_TYPE_META(T, idx) detail::TypeMetaData(idx, sizeof(T), STR_IMP(T)),
        FORALL_SCALAR_TYPES(SCALAR_TYPE_META)
    };
#undef SCALAR_TYPE_META
    return instances;
};

FORALL_SCALAR_TYPES(DELECARE_META_DATA);

} // namespace mariana 
