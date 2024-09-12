/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : tensor.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-17:20:10:14
 * Description:
 *
 */

#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <memory>
#include <core/tensor_impl.h>

namespace mariana {

class Tensor final {
public:
    Tensor(DataOn device=DataOn::CPU) {
        m_tensor = std::make_shared<TensorImpl>(device);
    }
    Tensor(const std::vector<int32_t>& dims, DataOn device=DataOn::CPU) {
        m_tensor = std::make_shared<TensorImpl>(dims, device);
    }
    Tensor(const std::vector<int32_t>& dims, DataOn device, void* data,
           TypeMeta dtype, bool move_data=false) {
        m_tensor = std::make_shared<TensorImpl>(dims, data, device, dtype, move_data);
    }
    Tensor(const Tensor&)=default;
    Tensor(Tensor&&)=default;
    ~Tensor()=default;
    DataOn device() const {
        return m_tensor->device();
    }
    TypeMeta dtype() const {
        return m_tensor->dtype();
    }
    std::string set_name() const {
        return m_tensor->name();
    }
    template <typename T>
    T data_at(uint64_t idx) const {
        return m_tensor->data_at<T>(idx);
    }
    template <typename T>
    T data(uint64_t idx) const {
        return m_tensor->data<T>(idx);
    }
    template <typename T>
    T* ptr(uint64_t idx) const {
        return m_tensor->ptr<T>(idx);
    }
    template <typename T>
    T* unsafe_ptr(uint64_t idx) const {
        return m_tensor->unsafe_ptr<T>(idx);
    }
    template <typename T>
    T* mutable_ptr() {
        return m_tensor->mutable_ptr<T>();
    }
    int32_t dim_at(size_t idx) const {
        return m_tensor->dim_at(idx);
    }
    std::vector<int32_t> dims() const {
        return m_tensor->dims();
    }
    int32_t stride_at(size_t idx) const {
        return m_tensor->stride_at(idx);
    }
    uint8_t dim_size() const {
        return m_tensor->dim_size();
    }
    uint32_t total_size() const {
        return m_tensor->total_size();
    }
    void set_name(const std::string& name) {
        m_tensor->set_name(name);
    }
    void set_dims(const std::vector<int32_t>& dims) {
        m_tensor->set_dims(dims);
    }
    void reshape(const std::vector<int32_t>& dims) {
        m_tensor->reshape(dims);
    }
    void try_realloc(const std::vector<int32_t>& dims, TypeMeta dtype) {
        m_tensor->try_realloc(dims, dtype);
    }
    Tensor& operator=(const Tensor& rhs) {
        if (this == &rhs) {
            return *this;
        }
        m_tensor = rhs.m_tensor;
        return *this;
    }
    Tensor deepcopy(void* extra=nullptr) const;
    Tensor shallowcopy() const;
    Tensor cuda(void* extra=nullptr) const;
    Tensor cpu(void* extra=nullptr) const;
private:
    std::shared_ptr<TensorImpl> m_tensor;
};

void to_bin(const Tensor& tensor);

void to_txt(const Tensor& tensor);

} // namespace mariana

#endif /* __TENSOR_H__ */
