/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/backend/gpu/cuda_common.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-04:15:49:27
 * Description:
 *
 */

#ifndef __CORE_BACKEND_GPU_CUDA_COMMON_H__
#define __CORE_BACKEND_GPU_CUDA_COMMON_H__

#include <string>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#define MLLM_CUDA_MAX_DEVICES       16
#define MLLM_CUDA_MAX_STREAMS       8

namespace mariana {

struct CUDAContext {
    int32_t device = -1;
    std::string name;
    cudaStream_t streams[MLLM_CUDA_MAX_STREAMS] = {nullptr};
    cublasHandle_t cublas_handles = nullptr;
    explicit CUDAContext(int32_t device) :
        device(device),
        name("cuda:"+std::to_string(device)) {
        cublas_handle();
        for (uint32_t i = 0; i < MLLM_CUDA_MAX_STREAMS; ++i) {
            stream(i);
        }
    }
    ~CUDAContext() {
        for (int32_t i = 0; i < MLLM_CUDA_MAX_STREAMS; ++i) {
            if (streams[i] != nullptr) {
                cudaStreamDestroy(streams[i]);
            }
        }
        if (cublas_handles != nullptr) {
            cublasDestroy(cublas_handles);
        }
    }
    void stream_sync(cudaStream_t stream);
    cudaStream_t stream() {
        return stream(0);
    }
    cudaStream_t stream(uint32_t i);
    cublasHandle_t cublas_handle();
};

struct CUDADeviceInfo {
    int32_t device_count = 0;
    struct DeviceInfo {
        int     cc;                 // compute capability
        int     nsm;                // number of streaming multiprocessors
        size_t  smpb;               // max. shared memory per block
        size_t  smpbo;              // max. shared memory per block (with opt-in)
        size_t  total_vram;
    };
    DeviceInfo devices[MLLM_CUDA_MAX_DEVICES] = {};
};

const CUDADeviceInfo& cuda_info();

int32_t cuda_get_device_count();

void cuda_get_device_memory(int32_t device, size_t* free, size_t* total);

void cuda_set_device(int32_t device);

int32_t cuda_get_device();

dim3 get_cuda_gridsize(size_t n, size_t cuda_blk);

} // namespace mariana

#endif /* __CORE_BACKEND_GPU_CUDA_COMMON_H__ */

