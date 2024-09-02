/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : graph.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-18:09:37:49
 * Description:
 * 
 */

#include <core/node.h>
#include <core/graph.h>
#include <ops/ops.h>
#include <utils/mariana_define.h>
#include <core/impl/thread_pool.h>

#include <mariana_llm/mariana_llm.h>

namespace mariana {

Graph::Graph(ExeContext& context) {
    int32_t n_th = context.n_threads == -1 ? ThreadPool::default_num_threads() : context.n_threads;
    m_tp = new ThreadPool(n_th);
}

NodeSharedPtr Graph::make_node(const OpCategory& opcate, const ModelParam& param, const std::vector<NodeSharedPtr>& inodes, const std::string& name) {
    NodeSharedPtr node = std::make_shared<Node>();
    bool ret = node->init(opcate, param, inodes, name);
    if (ret == false) {
        MLOG(ERROR)<<"Node:"<<name<<" init failed!";
        return nullptr;
    }
    node->set_thread_pool(m_tp);
    m_nodes.push_back(node);
    return node;
}

NodeSharedPtr Graph::make_root(const ModelParam& param, const std::string& name) {
    NodeSharedPtr node = std::make_shared<Node>();
    bool ret = node->init(OpCategory::Pass, param, {}, name);
    if (ret == false) {
        MLOG(ERROR)<<"Node:"<<name<<" init failed!";
        return nullptr;
    }
    node->set_thread_pool(m_tp);
    m_roots.push_back(node);
    return node;
}

NodeSharedPtr Graph::make_leaf(const OpCategory& opcate, const ModelParam& param, const std::vector<NodeSharedPtr>& inodes, const std::string& name) {
    NodeSharedPtr node = std::make_shared<Node>();
    bool ret = node->init(opcate, param, inodes, name);
    if (ret == false) {
        MLOG(ERROR)<<"Node:"<<name<<" init failed!";
        return nullptr;
    }
    node->set_thread_pool(m_tp);
    m_leafs.push_back(node);
    return node;
}

tensor_list Graph::forward(const KeyTensorMap& input_map, ExeContext& context) {
    TRACE();
    AUTOTIME("GRAPH RUN");
    for (size_t i = 0; i < m_roots.size(); ++i) {
        MCHECK_NE(input_map.count(m_roots[i]->name()), static_cast<size_t>(0))
            <<"Can't find input node tensor: "<<m_roots[i]->name();
        m_roots[i]->forward(input_map.at(m_roots[i]->name()), context);
    }
    for (size_t i = 0; i < m_nodes.size(); ++i) {
        tensor_list itensors;
        for(auto& inode : m_nodes[i]->inodes()) {
            itensors.insert(itensors.end(), inode->otensors().begin(), inode->otensors().end());
        }
        m_nodes[i]->forward(itensors, context);
    }
    tensor_list otensors;
    for (size_t i = 0; i < m_leafs.size(); ++i) {
        tensor_list itensors;
        for(auto& inode : m_leafs[i]->inodes()) {
            itensors.insert(itensors.end(), inode->otensors().begin(), inode->otensors().end());
        }
        m_leafs[i]->forward(itensors, context);
        otensors.insert(otensors.end(), m_leafs[i]->otensors().begin(), m_leafs[i]->otensors().end());
    }
    return otensors;
}

Graph::~Graph() {
    if (m_tp != nullptr) {
        delete m_tp;
    }
}

} // namespace mariana
