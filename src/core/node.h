/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/node.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-18:06:19:06
 * Description:
 *
 */

#ifndef __CORE_NODE_H__
#define __CORE_NODE_H__

#include <string>
#include <vector>
#include <memory>

#include <core/tensor.h>
#include <core/device_type.h>

namespace mariana {

class ThreadPool;
struct ExeContext;
using tensor_list = std::vector<Tensor>;
struct Function;
struct ModelParam;
enum class OpCategory : int16_t;

class Node final {
    using NodeSharedPtr = std::shared_ptr<Node>;
public:
    Node() {}
    ~Node() {}
    std::string name() const {
        return m_name;
    }
    DataOn device() const {
        return m_backend;
    }
    int8_t device_id() const {
        return m_device_id;
    }
    tensor_list otensors() const {
        return m_otensors;
    }
    tensor_list& otensors() {
        return m_otensors;
    }
    std::vector<NodeSharedPtr> inodes() const {
        return m_ins;
    }
    
    void set_device(const DataOn& device) {
        m_backend = device;
    }
    void set_device_id(int8_t device_id) {
        m_device_id = device_id;
    }
    void set_thread_pool(ThreadPool* tp);
    void forward(const tensor_list& inputs, ExeContext& context); // For input nodes run.
    bool plan_forward(const tensor_list& inputs, ExeContext& context);
    bool init(const OpCategory& opcate, const ModelParam& param, const std::vector<NodeSharedPtr>& inodes, const std::string& name);
    
private:
    std::string                m_name;
    tensor_list                m_otensors;
    OpCategory                 m_opcate;
    DataOn                     m_backend;
    int8_t                     m_device_id = -1;
    std::shared_ptr<Function>  m_op;
    std::vector<NodeSharedPtr> m_ins;
};

} // namespace mariana

#endif /* __CORE_NODE_H__ */
