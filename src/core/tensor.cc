/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : tensor.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-12:17:15:19
 * Description:
 * 
 */

#include <core/tensor.h>

#if defined(MLM_USE_CUDA)
#include <core/backend/gpu/cuda_allocator.h>
#endif

namespace mariana {

Tensor Tensor::shallowcopy() const {
    Tensor ret(dims(), device(), m_tensor->m_data, dtype());
    return ret;
}

Tensor Tensor::deepcopy(void* extra) const {
    std::vector<int32_t> dims;
    dims.resize(dim_size());
    for (size_t i = 0; i < dim_size(); ++i) {
        dims[i] = dim_at(i);
    }
    Tensor ret;
    if (device() == DataOn::CPU) {
        std::shared_ptr<TensorImpl> _tensor = std::make_shared<TensorImpl>(dims, device());
        IAllocator* allocator = get_allocator(_tensor->device());
        _tensor->m_dtype = m_tensor->m_dtype;
        _tensor->m_data = allocator->alloc(_tensor->m_total_size*_tensor->m_dtype.itemsize());
        allocator->memcpy(_tensor->m_data, m_tensor->m_data, _tensor->m_total_size*_tensor->m_dtype.itemsize());
        _tensor->m_own_data = true;
        ret.m_tensor = _tensor;
    } else if (device() == DataOn::GPU) {
#if defined(MLM_USE_CUDA)
        std::shared_ptr<TensorImpl> _tensor = std::make_shared<TensorImpl>(dims, device());
        IAllocator* allocator = get_allocator(_tensor->device());
        _tensor->m_dtype = m_tensor->m_dtype;
        _tensor->m_data = allocator->alloc(_tensor->m_total_size*_tensor->m_dtype.itemsize());
        CudaIAllocator::CudaMemcoryContext cmc;
        if (extra == nullptr) {
            cmc.stream = 0;
        } else {
            cmc.stream = static_cast<cudaStream_t>(extra);
        }
        cmc.kind = cudaMemcpyDeviceToDevice;
        allocator->memcpy(_tensor->m_data, m_tensor->m_data, _tensor->m_total_size*_tensor->m_dtype.itemsize(), &cmc);
        _tensor->m_own_data = true;
        ret.m_tensor = _tensor;
#else
        MLOG(FATAL)<<"Mariana_llm is not cimpiled with CUDA";
#endif
    } else {
        MLOG(FATAL)<<"Uninit tensor device: "<<device_string(device());
    }
    return ret;
}

Tensor Tensor::cuda(void* extra) const {
#if defined(MLM_USE_CUDA)
    if (device() == DataOn::GPU) {
        return *this;
    } else if (device() == DataOn::CPU) {
        Tensor ret;
        std::vector<int32_t> dims;
        dims.resize(dim_size());
        for (size_t i = 0; i < dim_size(); ++i) {
            dims[i] = dim_at(i);
        }
        std::shared_ptr<TensorImpl> _tensor = std::make_shared<TensorImpl>(dims, DataOn::GPU);
        IAllocator* allocator = get_allocator(DataOn::GPU);
        _tensor->m_dtype = m_tensor->m_dtype;
        _tensor->m_data = allocator->alloc(_tensor->m_total_size*_tensor->m_dtype.itemsize());
        CudaIAllocator::CudaMemcoryContext cmc;
        if (extra == nullptr) {
            cmc.stream = 0;
        } else {
            cmc.stream = static_cast<cudaStream_t>(extra);
        }
        cmc.kind = cudaMemcpyHostToDevice;
        allocator->memcpy(_tensor->m_data, m_tensor->m_data, _tensor->m_total_size*_tensor->m_dtype.itemsize(), &cmc);
        _tensor->m_own_data = true;
        ret.m_tensor = _tensor;
        return ret;
    } else {
        MLOG(FATAL)<<"Uninit tensor device: "<<device_string(device());
    }
#else
    MLOG(FATAL)<<"Mariana_llm is not cimpiled with CUDA";
#endif
}

Tensor Tensor::cpu(void* extra) const {
    if (device() == DataOn::CPU) {
        return *this;
    } else if (device() == DataOn::GPU) {
#if defined(MLM_USE_CUDA)
        Tensor ret;
        std::vector<int32_t> dims;
        dims.resize(dim_size());
        for (size_t i = 0; i < dim_size(); ++i) {
            dims[i] = dim_at(i);
        }
        std::shared_ptr<TensorImpl> _tensor = std::make_shared<TensorImpl>(dims, DataOn::CPU);
        IAllocator* allocator = get_allocator(DataOn::GPU);
        _tensor->m_dtype = m_tensor->m_dtype;
        _tensor->m_data = get_allocator(DataOn::CPU)->alloc(_tensor->m_total_size*_tensor->m_dtype.itemsize());
        CudaIAllocator::CudaMemcoryContext cmc;
        if (extra == nullptr) {
            cmc.stream = 0;
        } else {
            cmc.stream = static_cast<cudaStream_t>(extra);
        }
        cmc.kind = cudaMemcpyDeviceToHost;
        allocator->memcpy(_tensor->m_data, m_tensor->m_data, _tensor->m_total_size*_tensor->m_dtype.itemsize(), &cmc);
        _tensor->m_own_data = true;
        ret.m_tensor = _tensor;
        return ret;
#else
    MLOG(FATAL)<<"Mariana_llm is not cimpiled with CUDA";
#endif
    } else {
        MLOG(FATAL)<<"Uninit tensor device: "<<device_string(device());
    }    
}

} // namespace mariana
