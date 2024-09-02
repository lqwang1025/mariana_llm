/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : cpu_alloc.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-13:07:00:08
 * Description:
 *
 */

#ifndef __CPU_ALLOC_H__
#define __CPU_ALLOC_H__

#include <cstddef>

namespace mariana {

void* alloc_cpu(size_t nbytes);

void free_cpu(void* data);

void memcpy_cpu(void* dst, const void* src, size_t nbytes);

void memset_cpu(void* dst, int c, size_t nbytes);

} // namespace mariana

#endif /* __CPU_ALLOC_H__ */

