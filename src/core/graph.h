/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/graph.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-18:09:33:42
 * Description:
 *
 */

#ifndef __CORE_GRAPH_H__
#define __CORE_GRAPH_H__

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <core/tensor.h>

namespace mariana {

struct ExeContext;
class Node;
class ThreadPool;
enum class OpCategory : int16_t;
using tensor_list = std::vector<Tensor>;
using NodeSharedPtr = std::shared_ptr<Node>;
using KeyTensorMap = std::unordered_map<std::string, tensor_list>;

struct ModelParam;

class Graph final {
public:
    Graph(int32_t num_of_threads);
    ~Graph();
    ThreadPool* thread_pool() const {
        return m_tp;
    }
    bool gpu_distribute();
    NodeSharedPtr make_root(const ModelParam& param, const std::string& name="");
    NodeSharedPtr make_node(const OpCategory& opcate, const ModelParam& param, const std::vector<NodeSharedPtr>& inodes, const std::string& name="");
    NodeSharedPtr make_leaf(const OpCategory& opcate, const ModelParam& param, const std::vector<NodeSharedPtr>& inodes, const std::string& name="");
    tensor_list forward(const KeyTensorMap& input_map, ExeContext& context);
private:
    ThreadPool* m_tp = nullptr;
    std::vector<NodeSharedPtr> m_nodes;
    std::vector<NodeSharedPtr> m_leafs;
    std::vector<NodeSharedPtr> m_roots;
    std::vector<NodeSharedPtr> m_all_nodes;
};
    
} // namespace mariana

#endif /* __CORE_GRAPH_H__ */

