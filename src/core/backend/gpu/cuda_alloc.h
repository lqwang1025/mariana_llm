/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : cuda_alloc.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-17:11:19:25
 * Description:
 *
 */

#ifndef __CUDA_ALLOC_H__
#define __CUDA_ALLOC_H__

namespace mariana {

void* alloc_cuda(size_t nbytes);

void free_cuda(void* data);

void* alloc_cuda_host(size_t nbytes);

void free_cuda_host(void* data);

} // namespace mariana

#endif /* __CUDA_ALLOC_H__ */

