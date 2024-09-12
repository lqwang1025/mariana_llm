/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/backend/backend.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-04:15:39:59
 * Description:
 * 
 */

#include <core/backend/backend.h>

#if defined(MLM_USE_CUDA)
#include <core/backend/gpu/cuda_common.h>
#endif

namespace mariana {

BackendContext::~BackendContext() {
    if (DataOn::GPU == device) {
        if (context != nullptr) {
            CUDAContext* cuda_context = static_cast<CUDAContext*>(context);
            delete cuda_context;
        }
    }
}
    
} // namespace mariana

