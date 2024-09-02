/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : cuda_alloc.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-17:11:19:41
 * Description:
 * 
 */

#include <cuda_runtime_api.h>
#include <utils/mariana_define.h>
#include <core/backend/gpu/cuda_alloc.h>
#include <core/backend/gpu/helper_cuda.h>

namespace mariana {

void* alloc_cuda(size_t nbytes) {
    if (nbytes == 0) {
        return nullptr;
    }
    MCHECK_GT(nbytes, static_cast<size_t>(0))<<__func__<<" seems to have been called with negative number:"<<nbytes;
    void *devPtr = nullptr;
    checkCudaErrors(cudaMalloc(&devPtr, nbytes));
    return devPtr;
}

void free_cuda(void* data) {
    if (data) {
        checkCudaErrors(cudaFree(data));
    }
    data = nullptr;
}

void* alloc_cuda_host(size_t nbytes) {
    if (nbytes == 0) {
        return nullptr;
    }
    MCHECK_GT(nbytes, static_cast<size_t>(0))<<__func__<<" seems to have been called with negative number:"<<nbytes;
    void *data = nullptr;
    checkCudaErrors(cudaMallocHost(&data, nbytes));
    return data;
}

void free_cuda_host(void* data) {
    if (data) {
        checkCudaErrors(cudaFreeHost(data));
    }
    data = nullptr;
}

} // namespace mariana
