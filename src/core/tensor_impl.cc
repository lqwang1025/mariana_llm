/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/tensor_impl.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-20:10:47:34
 * Description:
 * 
 */

#include <core/tensor_impl.h>
#include <core/impl/allocator.h>

namespace mariana {

void TensorImpl::_try_alloc(TypeMeta dtype) {
    TRACE();
    if (m_data == nullptr) {
        IAllocator* allocator = get_allocator(m_device);
        m_dtype = dtype;
        m_data = allocator->alloc(m_total_size*m_dtype.itemsize());
        m_own_data = true;
    }
}

void TensorImpl::_try_destory() {
    TRACE();
    if (m_data !=nullptr && m_own_data) {
        IAllocator* allocator = get_allocator(m_device);
        allocator->free(m_data);
    }
    m_own_data = false;
    m_data = nullptr;
}

bool TensorImpl::reshape(const std::vector<int32_t>& dims) {
    uint32_t _new_total_size = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
        _new_total_size *= dims[i];
    }
    bool ok = true;
    ok = ok && (_new_total_size == m_total_size);
    if (ok) {
        set_dims(dims);
    } else {
        MLOG(ERROR)<<"The total number of elements needs to be equal "
                   <<"before and after the tensor reshape: now "
                   <<_new_total_size <<" "<<m_total_size;
    }
    return ok;
}

void TensorImpl::set_dims(const std::vector<int32_t>& dims) {
    TRACE();
    MCHECK_LE(dims.size(), MAX_DIM_SIZE);
    m_dim_size = dims.size();
    m_total_size = 1;
    for (size_t i = 0; i < m_dim_size; ++i) {
        m_dims[i] = dims[i];
        m_total_size *= m_dims[i];
    }
    for (size_t i = 0; i < m_dim_size; ++i) {
        int32_t stride = 1;
        for (size_t j = i+1; j < dim_size(); ++j) {
            stride *= dim_at(j);
        }
        m_strides[i] = stride;
    }
}

void TensorImpl::try_realloc(const std::vector<int32_t>& dims, TypeMeta dtype) {
    TRACE();
    MCHECK_LE(dims.size(), MAX_DIM_SIZE);
    m_dim_size = dims.size();
    int32_t new_total_size = 1;
    for (size_t i = 0; i < m_dim_size; ++i) {
        m_dims[i] = dims[i];
        new_total_size *= m_dims[i];
    }
    if (m_total_size*m_dtype.itemsize() < new_total_size*dtype.itemsize()) {
        m_total_size = new_total_size;
        _try_destory();
        _try_alloc(dtype);
    }
    m_total_size = new_total_size;
    m_dtype = dtype;
    for (size_t i = 0; i < m_dim_size; ++i) {
        int32_t stride = 1;
        for (size_t j = i+1; j < dim_size(); ++j) {
            stride *= dim_at(j);
        }
        m_strides[i] = stride;
    }
}

} // namespace mariana
