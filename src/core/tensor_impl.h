/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/tensor_impl.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-12:17:15:14
 * Description:
 *
 */

#ifndef __CORE_TENSOR_IMPL_H__
#define __CORE_TENSOR_IMPL_H__

#include <core/data_type.h>
#include <core/device_type.h>
#include <utils/mariana_define.h>

namespace mariana {

class TensorImpl {
public:
    TensorImpl(DataOn device=DataOn::CPU) : m_device(device) {}
    TensorImpl(const std::vector<int32_t>& dims, DataOn device=DataOn::CPU) : TensorImpl(device) {
        set_dims(dims);
    }
    TensorImpl(const std::vector<int32_t>& dims, void* data, DataOn device,
               TypeMeta dtype, bool move_data) {
        set_dims(dims);
        m_data     = data;
        m_dtype    = dtype;
        m_device   = device;
        m_own_data = move_data;
    }
    
    ~TensorImpl() {
        _try_destory();
    }    
    DataOn device() const {
        return m_device;
    }
    TypeMeta dtype() const {
        return m_dtype;
    }
    std::string name() const {
        return m_name;
    }
    template <typename T>
    T data_at(uint64_t idx) const {
        return *unsafe_ptr<T>(idx);
    }

    template <typename T>
    T data(uint64_t idx) const {
        return *ptr<T>(idx);
    }
    
    template <typename T>
    T* ptr(uint64_t idx) const {
        if (idx > m_total_size) {
            MLOG(ERROR)<<idx<<" Out of tensor range of:"<<m_total_size;
            return nullptr;
        }
        if (false == m_dtype.match<T>()) {
            MLOG(WARNING)<<"Tensor dtype mismatch tensor:"<<m_dtype.name()
                         <<"<->"<<TypeMeta::make<T>().name();
        }
        return static_cast<T*>(m_data)+idx;
    }
    
    template <typename T>
    T* unsafe_ptr(uint64_t idx) const {
        MLOG_IF(FATAL, idx >= m_total_size)<<"Out of range error, size:"
                                           <<m_total_size<<"<="<<idx;
        return static_cast<T*>(m_data)+idx;
    }
    template <typename T>
    T* mutable_ptr() {
        return _raw_ptr<T>();
    }
    int32_t dim_at(size_t idx) const {
        MCHECK_GE(idx, static_cast<size_t>(0));
        MCHECK_LT(idx, dim_size());
        return m_dims[idx];
    }
    int32_t stride_at(size_t idx) const {
        MCHECK_GE(idx, static_cast<size_t>(0));
        MCHECK_LT(idx, dim_size());
        return m_strides[idx];
    }
    uint8_t dim_size() const {
        return m_dim_size;
    }
    std::vector<int32_t> dims() const {
        std::vector<int32_t> dims;
        dims.reserve(dim_size());
        for (uint8_t i = 0; i < dim_size(); ++i) {
            dims.push_back(dim_at(i));
        }
        return dims;
    }
    uint32_t total_size() const {
        return m_total_size;
    }
    void set_name(const std::string& name) {
        m_name = name;
    }
    void set_dims(const std::vector<int32_t>& dims);
    bool reshape(const std::vector<int32_t>& dims);
    void try_realloc(const std::vector<int32_t>& dims, TypeMeta dtype);
private:
    template<typename T>
    T* _raw_ptr() {
        TRACE();
        if (m_total_size == 0) {
            MLOG(WARNING)<<"TensorImpl data pointer will return nullptr because of:"
                         <<"[all dim is zero]";
            return nullptr;
        }
        if (m_data != nullptr && m_dtype.match<T>() == false) {
            MLOG(WARNING)<<"TensorImpl dtype is:"<<m_dtype.name()
                         <<" but you want dtype:"<<TypeMeta::make<T>().name();
        } else {
            _try_alloc(TypeMeta::make<T>());
        }
        return static_cast<T*>(m_data);
    }
    
    void _try_destory();
    void _try_alloc(TypeMeta dtype);
    static constexpr uint8_t MAX_DIM_SIZE = 6;
    friend class Tensor;
private:
    int32_t     m_dims[MAX_DIM_SIZE]    = {-1, -1, -1, -1, -1, -1};
    int32_t     m_strides[MAX_DIM_SIZE] = {-1, -1, -1, -1, -1, -1};
    uint8_t     m_dim_size              = 0;
    uint32_t    m_total_size            = 0;
    bool        m_own_data              = false;
    void*       m_data                  = nullptr;    
    DataOn      m_device                = DataOn::CPU;
    std::string m_name                  = "";
    TypeMeta m_dtype;
    DISABLE_COPY_AND_ASSIGN(TensorImpl);
};
    
} // namespace mariana

#endif /* __CORE_TENSOR_IMPL_H__ */

