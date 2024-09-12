/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : allocator.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-17:14:01:49
 * Description:
 * 
 */

#include <cuda_runtime_api.h>
#include <utils/mariana_define.h>
#include <core/device_type.h>
#include <core/impl/cpu_alloc.h>
#include <core/backend/gpu/helper_cuda.h>
#include <core/backend/gpu/cuda_allocator.h>
#include <core/backend/gpu/cuda_alloc.h>

namespace mariana {

void* CudaIAllocator::m_alloc_impl(size_t size) {
    return alloc_cuda(size);
}

void CudaIAllocator::m_free_impl(void* data) {
    free_cuda(data);
}

void CudaIAllocator::_memcpy(void* dst, const void* src, size_t size, CudaMemcoryContext cmc) {
    if (cmc.sync) {
        checkCudaErrors(cudaMemcpy(dst, src, size, cmc.kind));
    } else {
        checkCudaErrors(cudaMemcpyAsync(dst, src, size, cmc.kind, cmc.stream));
    }
}

void CudaIAllocator::memcpy(void* dst, const void* src, size_t size, void* extra) {
    MCHECK_NOTNULL(extra);
    CudaMemcoryContext* cmc = static_cast<CudaMemcoryContext*>(extra);
    _memcpy(dst, src, size, *cmc);
}

void CudaIAllocator::memset(void* dst, int32_t c, size_t size, void* extra) {
    // MCHECK_NOTNULL(extra);
    // CudaMemcoryContext* cmc = static_cast<CudaMemcoryContext*>(extra);
    checkCudaErrors(cudaMemset(dst, c, size));
}

void* CudaHostIAllocator::m_alloc_impl(size_t size) {
    return alloc_cuda_host(size);
}

void CudaHostIAllocator::m_free_impl(void* data) {
    free_cuda_host(data);
}

} // namespace mariana
