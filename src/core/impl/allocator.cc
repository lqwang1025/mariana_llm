/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : allocator.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-17:12:51:09
 * Description:
 * 
 */


#include <core/device_type.h>
#include <core/impl/allocator.h>
#include <core/impl/cpu_alloc.h>
#include <utils/mariana_define.h>

namespace mariana {

void AllocatorStat::alloc_record(size_t size, void* data) {
    alloc_count += size;
    if (static_cast<size_t>(max) < size) {
        max = size;
    }
    MVLOG(2)<<"Alloc [size]:"<<size<<" data pointer:"<<data;
    pointer_map[data] = size;
}

void AllocatorStat::free_record(void* data) {
    int32_t size = pointer_map.at(data);
    pointer_map.erase(data);
    MVLOG(2)<<"Free [size]:"<<size<<" data pointer:"<<data;
    free_count += size;
}

void AllocatorStat::summary() const {
    MLOG(INFO)<<"Memory stat info: alloc total [size]: "<<alloc_count<<" \n"
              <<" free total [size]: "<<free_count<<" \n"
              <<" max chunk [size]: "<<max;
}

void CpuIAllocator::memset(void* dst, int32_t c, size_t size, void* extra) {
    memset_cpu(dst, c, size);
}

void CpuIAllocator::memcpy(void* dst, const void* src, size_t size, void* extra) {
    memcpy_cpu(dst, src, size);
}
 
void* CpuIAllocator::m_alloc_impl(size_t size) {
    return alloc_cpu(size);
}

void CpuIAllocator::m_free_impl(void* data) {
    free_cpu(data);
}

inline static std::unordered_map<DataOn, IAllocator*>& _get_allocator_map() {
    static std::unordered_map<DataOn, IAllocator*> allocator_map;
    return allocator_map;
}

IAllocator* get_allocator(const DataOn& device) {
    std::unordered_map<DataOn, IAllocator*>& allocator_map = _get_allocator_map();
    MLOG_IF(ERROR, allocator_map.count(device)==0)<<"Device: "<<device_string(device)<<" has no allocator!";
    return allocator_map[device];
}

void set_allocator(IAllocator* allocator, const DataOn& device) {
    std::unordered_map<DataOn, IAllocator*>& allocator_map = _get_allocator_map();
    allocator_map[device] = allocator;
}

} // namespace mariana
