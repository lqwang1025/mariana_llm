/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/impl/allocator.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-13:08:54:02
 * Description:
 *
 */

#ifndef __ALLOCATOR_H__
#define __ALLOCATOR_H__

#include <cstddef>
#include <unordered_map>

#include <core/device_type.h>

namespace mariana {

struct AllocatorStat {
    int32_t alloc_count = 0;
    int32_t free_count = 0;
    int32_t max = 0;
    std::unordered_map<void*, int32_t> pointer_map;
    void alloc_record(size_t size, void* data);
    void free_record(void* data);
    void summary() const;
};

class IAllocator {
public:
    IAllocator() {}
    virtual ~IAllocator() = default;
    void* alloc(size_t size) {
        void* data = m_alloc_impl(size);
        stat_info.alloc_record(size, data);
        return data;
    }
    void free(void* data) {
        stat_info.free_record(data);
        m_free_impl(data);
    }
    void summary() const {
        stat_info.summary();
    }
    virtual void memcpy(void* dst, const void* src, size_t size, void* extra=nullptr)=0;
    virtual void memset(void* dst, int32_t c, size_t size, void* extra=nullptr)=0;
protected:
    virtual void* m_alloc_impl(size_t size)=0;
    virtual void m_free_impl(void* data)=0;
    AllocatorStat stat_info;
};

class CpuIAllocator : public IAllocator {
public:
    CpuIAllocator() {}
    virtual ~CpuIAllocator() {}
    virtual void memcpy(void* dst, const void* src, size_t size, void* extra=nullptr) override;
    virtual void memset(void* dst, int32_t c, size_t size, void* extra=nullptr) override;
protected:
    virtual void* m_alloc_impl(size_t size) override;
    virtual void m_free_impl(void* data) override;
};

IAllocator* get_allocator(const DataOn& device);

void set_allocator(IAllocator* allocator, const DataOn& device);

template <DataOn type>
struct AllocatorRegisterer {
    explicit AllocatorRegisterer(IAllocator* alloc) {
        set_allocator(alloc, type);
    }
};

#define REGISTER_ALLOCATOR(t, f)                        \
    static AllocatorRegisterer<t> g_allocator_d_##f(new f)

} // namespace mariana

#endif /* __ALLOCATOR_H__ */

