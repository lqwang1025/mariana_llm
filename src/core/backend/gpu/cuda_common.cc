/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/backend/gpu/cuda_common.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-04:15:50:08
 * Description:
 * 
 */

#include <cmath>
#include <cuda_runtime.h>

#include <utils/mariana_define.h>

#include <core/backend/gpu/cuda_common.h>
#include <core/backend/gpu/helper_cuda.h>

namespace mariana {

void CUDAContext::stream_sync(cudaStream_t stream) {
    checkCudaErrors(cudaStreamSynchronize(stream));
}

cudaStream_t CUDAContext::stream(uint32_t i) {
    uint32_t __i = i%MLLM_CUDA_MAX_STREAMS;
    if (streams[__i] == nullptr) {
        cuda_set_device(device);
        checkCudaErrors(cudaStreamCreateWithFlags(&streams[__i], cudaStreamNonBlocking));
    }
    return streams[__i];
}

cublasHandle_t CUDAContext::cublas_handle() {
    if (cublas_handles == nullptr) {
        cuda_set_device(device);
        checkCudaErrors(cublasCreate(&cublas_handles));
        checkCudaErrors(cublasSetMathMode(cublas_handles, CUBLAS_TF32_TENSOR_OP_MATH));
    }
    return cublas_handles;
}

static CUDADeviceInfo cuda_init() {
    CUDADeviceInfo info = {};
    cudaError_t err = cudaGetDeviceCount(&info.device_count);
    checkCudaErrors(err);
    MLOG_IF(FATAL, MLLM_CUDA_MAX_DEVICES < info.device_count)<<
        "max cuda size:"<<MLLM_CUDA_MAX_DEVICES<<" now:"<<info.device_count;

    for (int32_t id = 0; id < info.device_count; ++id) {
        cuda_set_device(id);
        cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties(&prop, id));
        MLOG_IF(FATAL, checkCudaCapabilities(prop.major, prop.minor)==false);
        MVLOG(2)<<"Device-"<<id<<":"<<prop.name<<", compute capability:"
                <<prop.major<<"."<<prop.minor;
        info.devices[id].nsm   = prop.multiProcessorCount;
        info.devices[id].smpb  = prop.sharedMemPerBlock;
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        info.devices[id].total_vram = prop.totalGlobalMem;
    }
    cuda_set_device(0);
    return info;
}

const CUDADeviceInfo& cuda_info() {
    static CUDADeviceInfo info = cuda_init();
    return info;
}

void cuda_get_device_memory(int32_t device, size_t* free, size_t* total) {
    cuda_set_device(device);
    checkCudaErrors(cudaMemGetInfo(free, total));
}

int32_t cuda_get_device_count() {
    return cuda_info().device_count;
}

void cuda_set_device(int32_t device) {
    int32_t current_device;
    checkCudaErrors(cudaGetDevice(&current_device));
    if (current_device == device) return;
    checkCudaErrors(cudaSetDevice(device));
}

int32_t cuda_get_device() {
    int32_t current_device;
    checkCudaErrors(cudaGetDevice(&current_device));
    return current_device;
}

dim3 get_cuda_gridsize(size_t n, size_t cuda_blk) {
    uint32_t k = (n-1) / cuda_blk + 1;
    uint32_t x = k;
    uint32_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*cuda_blk) + 1;
    }
    dim3 d = {x, y, 1};
    return d;
}

} // namespace mariana
