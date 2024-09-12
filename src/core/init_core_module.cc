/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : init_core_module.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-17:21:37:02
 * Description:
 * 
 */

#include <core/impl/allocator.h>
#include <core/device_type.h>
#include <core/init_core_module.h>

#if defined(MLM_USE_CUDA)
#include <core/backend/gpu/cuda_allocator.h>
#endif

namespace mariana {

void init_core_module() {
    REGISTER_ALLOCATOR(DataOn::CPU, CpuIAllocator);
#if defined(MLM_USE_CUDA)
    REGISTER_ALLOCATOR(DataOn::GPU, CudaIAllocator);
    //REGISTER_ALLOCATOR(DataOn::CPU, CudaHostIAllocator);
#endif
}

} // namespace mariana
