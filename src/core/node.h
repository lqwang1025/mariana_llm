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
#include <core/function.h>
#include <core/device_type.h>
#include <core/backend/backend.h>

#include <absl/synchronization/mutex.h>

namespace mariana {

class ThreadPool;
struct ExeContext;
using tensor_list = std::vector<Tensor>;
struct Function;
struct ModelParam;
enum class OpCategory : int16_t;

struct RuntimeInfo {
    uint32_t feature_height = 0;
    uint32_t feature_width  = 0;
    void* anything = nullptr;
};

class Node final {
    using NodeSharedPtr = std::shared_ptr<Node>;
public:
    Node() : m_complete(false) {}
    ~Node();
    std::string name() const {
        return m_name;
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
    void set_inputs(const tensor_list& inputs) {
        m_complete = false;
        m_itensors = inputs;
    }
    void push_info_shared_nodes(const std::vector<NodeSharedPtr>& nodes) {
        m_info_shared_nodes.insert(m_info_shared_nodes.end(), nodes.begin(), nodes.end());
    }
    const std::vector<NodeSharedPtr>& info_shared_nodes() const {
        return m_info_shared_nodes;
    }
    RuntimeInfo runtime_info() const {
        return m_runtime_info;
    }
    RuntimeInfo& runtime_info() {
        return m_runtime_info;
    }
    void setup_backend(std::shared_ptr<BackendContext> backend_ctx) {
        m_backend_ctx = backend_ctx;
    }
    std::shared_ptr<BackendContext> backend_ctx() const {
        return m_backend_ctx;
    }
    void wait_for_done();
    void set_thread_pool(size_t num_of_threads);
    void forward(ExeContext& context); // For input nodes run.
    bool init(const OpCategory& opcate, const ModelParam& param, const std::vector<NodeSharedPtr>& inodes, const std::string& name);
private:
    std::string                     m_name;
    tensor_list                     m_otensors;
    tensor_list                     m_itensors;
    OpCategory                      m_opcate;
    ThreadPool*                     m_tp;
    std::shared_ptr<BackendContext> m_backend_ctx;
    std::unique_ptr<Function>       m_op;
    std::vector<NodeSharedPtr>      m_ins;
    std::vector<NodeSharedPtr>      m_info_shared_nodes;
    absl::Mutex                     m_mutex;
    bool                            m_complete;
    RuntimeInfo                     m_runtime_info;
};

} // namespace mariana

#endif /* __CORE_NODE_H__ */
