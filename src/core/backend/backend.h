/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/backend/backend.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-04:15:39:54
 * Description:
 *
 */

#ifndef __CORE_BACKEND_BACKEND_H__
#define __CORE_BACKEND_BACKEND_H__

#include <mariana_llm/mariana_llm_impl.h>

namespace mariana {

struct BackendContext {
    explicit BackendContext(DataOn device, void* context) :
        device(device), context(context) {}
    ~BackendContext();
    DataOn device = DataOn::CPU;
    void* context = nullptr;
};
    
} // namespace mariana

#endif /* __CORE_BACKEND_BACKEND_H__ */

