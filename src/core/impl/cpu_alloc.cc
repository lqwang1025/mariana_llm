/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : cpu_alloc.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-13:07:00:14
 * Description:
 * 
 */

#include <cstring>
#include <core/impl/cpu_alloc.h>
#include <utils/mariana_define.h>

namespace mariana {

static void* aligned_malloc(size_t nbytes, size_t alignment) {
    int offset = alignment - 1 + sizeof(void*);
    void* ptr = (void*)malloc(nbytes+offset);
    MCHECK_NOTNULL(ptr);
    void** p2 = (void**)(((size_t)ptr+offset) & ~(alignment -1));
    p2[-1] = ptr;
    return p2;
}

static void aligned_free(void* data) {
    void* ptr = ((void**)data)[-1];
    free(ptr);
}

void* alloc_cpu(size_t nbytes) {
    if (nbytes == 0) {
        return nullptr;
    }
    MCHECK_GT(nbytes, static_cast<size_t>(0))<<__func__<<" seems to have been called with negative number:"<<nbytes;
    const size_t alignment = 16;
    return aligned_malloc(nbytes, alignment);
}

void free_cpu(void* data) {
    aligned_free(data);
}

void memcpy_cpu(void* dst, const void* src, size_t nbytes) {
    memcpy(dst, src, nbytes);
}

void memset_cpu(void* dst, int c, size_t nbytes) {
    memset(dst, c, nbytes);
}

} // namespace mariana
