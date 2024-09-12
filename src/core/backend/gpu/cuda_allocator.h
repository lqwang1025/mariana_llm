/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/backend/gpu/cuda_allocator.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-17:14:00:31
 * Description:
 *
 */

#ifndef __CUDA_ALLOCATOR_H__
#define __CUDA_ALLOCATOR_H__

#include <atomic>
#include <cuda_runtime_api.h>
#include <core/impl/allocator.h>

namespace mariana {

class CudaIAllocator : public mariana::IAllocator {
public:
    CudaIAllocator() {}
    ~CudaIAllocator() {}
    struct CudaMemcoryContext {
        cudaStream_t stream = 0;
        bool         sync   = false;
        cudaMemcpyKind kind;
        
    };
    virtual void memcpy(void* dst, const void* src, size_t size, void* extra=nullptr)override;
    virtual void memset(void* dst, int32_t c, size_t size, void* extra=nullptr) override;
protected:
    void _memcpy(void* dst, const void* src, size_t size, CudaMemcoryContext cmc);
    void* m_alloc_impl(size_t size) override;
    void m_free_impl(void* data) override;
};

class CudaHostIAllocator : public CpuIAllocator {
public:
    CudaHostIAllocator() {}
    ~CudaHostIAllocator() {}
protected:
    void* m_alloc_impl(size_t size) override;
    void m_free_impl(void* data) override;
};

} // namespace mariana

#endif /* __CUDA_ALLOCATOR_H__ */

