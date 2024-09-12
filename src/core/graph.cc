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

#include <ops/ops.h>

#include <core/node.h>
#include <core/graph.h>
#include <core/impl/thread_pool.h>

#include <utils/progress_bar.h>
#include <utils/mariana_define.h>

#include <mariana_llm/mariana_llm.h>

#if defined(MLM_USE_CUDA)
#include <core/backend/gpu/cuda_common.h>
#endif

namespace mariana {

Graph::Graph(int32_t num_of_threads) {
    int32_t n_th = num_of_threads == -1 ? ThreadPool::default_num_threads() : num_of_threads;
    m_tp = new ThreadPool(n_th);
}

NodeSharedPtr Graph::make_node(const OpCategory& opcate, const ModelParam& param, const std::vector<NodeSharedPtr>& inodes, const std::string& name) {
    NodeSharedPtr node = std::make_shared<Node>();
    bool ret = node->init(opcate, param, inodes, name);
    if (ret == false) {
        MLOG(ERROR)<<"Node:"<<name<<" init failed!";
        return nullptr;
    }
    node->set_thread_pool(m_tp->size());
    m_nodes.push_back(node);
    m_all_nodes.push_back(node);
    return node;
}

NodeSharedPtr Graph::make_root(const ModelParam& param, const std::string& name) {
    NodeSharedPtr node = std::make_shared<Node>();
    bool ret = node->init(OpCategory::Pass, param, {}, name);
    if (ret == false) {
        MLOG(ERROR)<<"Node:"<<name<<" init failed!";
        return nullptr;
    }
    node->set_thread_pool(m_tp->size());
    m_roots.push_back(node);
    m_all_nodes.push_back(node);
    return node;
}

NodeSharedPtr Graph::make_leaf(const OpCategory& opcate, const ModelParam& param, const std::vector<NodeSharedPtr>& inodes, const std::string& name) {
    NodeSharedPtr node = std::make_shared<Node>();
    bool ret = node->init(opcate, param, inodes, name);
    if (ret == false) {
        MLOG(ERROR)<<"Node:"<<name<<" init failed!";
        return nullptr;
    }
    node->set_thread_pool(m_tp->size());
    m_leafs.push_back(node);
    m_all_nodes.push_back(node);
    return node;
}

tensor_list Graph::forward(const KeyTensorMap& input_map, ExeContext& context) {
    TRACE();
    ProgressBar pb("Generate:");
    float _total   = m_roots.size()+m_nodes.size()+m_leafs.size();
    float _current = 0;
    for (size_t i = 0; i < m_roots.size(); ++i) {
        MCHECK_NE(input_map.count(m_roots[i]->name()), static_cast<size_t>(0))
            <<"Can't find input node tensor: "<<m_roots[i]->name();
        pb.print_bar("", static_cast<uint16_t>(std::round(_current/_total*100.f)));
        _current += 1;
        m_roots[i]->set_inputs(input_map.at(m_roots[i]->name()));
        m_tp->submit(std::mem_fn(&Node::forward), m_roots[i].get(), std::ref(context));
    }
    for (size_t i = 0; i < m_nodes.size(); ++i) {
        tensor_list itensors;
        for(auto& inode : m_nodes[i]->inodes()) {
            inode->wait_for_done();
            itensors.insert(itensors.end(), inode->otensors().begin(), inode->otensors().end());
        }
        pb.print_bar("", static_cast<uint16_t>(std::round(_current/_total*100.f)));
        _current += 1;
        m_nodes[i]->set_inputs(itensors);
        m_tp->submit(std::mem_fn(&Node::forward), m_nodes[i].get(), std::ref(context));
    }
    for (size_t i = 0; i < m_leafs.size(); ++i) {
        tensor_list itensors;
        for(auto& inode : m_leafs[i]->inodes()) {
            inode->wait_for_done();
            itensors.insert(itensors.end(), inode->otensors().begin(), inode->otensors().end());
        }
        pb.print_bar("", static_cast<uint16_t>(std::round(_current/_total*100.f)));
        _current += 1;
        m_leafs[i]->set_inputs(itensors);
        m_tp->submit(std::mem_fn(&Node::forward), m_leafs[i].get(), std::ref(context));
    }
    
    tensor_list otensors;
    for (size_t i = 0; i < m_leafs.size(); ++i) {
        m_leafs[i]->wait_for_done();
        otensors.insert(otensors.end(), m_leafs[i]->otensors().begin(), m_leafs[i]->otensors().end());
    }
    return otensors;
}

bool Graph::gpu_distribute() {
#if defined(MLM_USE_CUDA)
    std::vector<float> splits;
    float split_sum = 0.0f;
    int32_t device_count = cuda_get_device_count();
    std::vector<std::shared_ptr<BackendContext>> backend_ctxes;
    for (int i = 0; i < device_count; ++i) {
        size_t free = 0, total = 0;
        cuda_get_device_memory(i, &free, &total);
        split_sum += free;
        splits.push_back(split_sum);
    }
    for (int i = 0; i < device_count; ++i) {
        splits[i] /= split_sum;
        CUDAContext* cuda_ctx = new CUDAContext{i};
        backend_ctxes.push_back(std::make_shared<BackendContext>(DataOn::GPU, cuda_ctx));
    }

    for (size_t i = 0; i < m_all_nodes.size(); ++i) {
        int layer_gpu = std::upper_bound(splits.begin(), splits.end(), float(i)/float(m_all_nodes.size())) - splits.begin();
        NodeSharedPtr node = m_all_nodes[i];
        node->setup_backend(backend_ctxes[layer_gpu]);
    }
    return true;
#else
    MLOG(ERROR)<<"Mariana_llm is not compiled with cuda!";
    return false;
#endif
}

Graph::~Graph() {
    if (m_tp != nullptr) {
        delete m_tp;
    }
}

} // namespace mariana
