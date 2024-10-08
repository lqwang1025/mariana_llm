/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : node.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-18:06:19:09
 * Description:
 * 
 */

#include <ops/ops.h>

#include <core/tensor_utils.h>
#include <core/node.h>
#include <core/function.h>

#include <utils/sys.h>
#include <utils/mariana_define.h>

namespace mariana {

Node::~Node() {
    if (m_tp != nullptr)
        delete m_tp;
}

void Node::set_thread_pool(size_t num_of_threads) {
    m_tp = new ThreadPool(num_of_threads);
    m_op->set_thread_pool(m_tp);
}

void Node::wait_for_done() {
    absl::MutexLock lck(&m_mutex);
    absl::Condition complete(&m_complete);
    m_mutex.Await(complete);
}

bool Node::init(const OpCategory& opcate, const ModelParam& param, const std::vector<NodeSharedPtr>& inodes, const std::string& name) {
    TRACE();
    m_opcate = opcate;
    m_name = name;
    auto func_maker = FunctionHolder::search(opcate);
    Function* op = func_maker();
    m_op.reset(op);
    m_op->set_node(this);
    m_ins = inodes;
    return m_op->init(param, name);
}

void Node::forward(ExeContext& context) {
    absl::MutexLock lck(&m_mutex);
    m_complete = false;
    MLOG_IF(ERROR, !m_op)<<"m_op null";
    {
        AUTOTIME(op_to_string(m_opcate).c_str());
        bool ok = m_op->on_plan_forward(m_itensors, m_otensors, context);
        ok = ok && m_op->on_forward(m_itensors, m_otensors, context);
        MLOG_IF(ERROR, !ok)<<"Node:"<<op_to_string(m_opcate)<<" forward failed";
    }
    if (get_env("MAR_LAYER_RES_DUMP") == "1") {
        for (auto& tensor : m_otensors) {
            if (tensor.device() == DataOn::GPU) {
                Tensor _tensor = tensor.cpu();
                DUMP_TENSOR_TO_BIN(_tensor, op_to_string(m_opcate));
                DUMP_TENSOR_TO_TXT(_tensor, op_to_string(m_opcate));
            } else {
                DUMP_TENSOR_TO_BIN(tensor, op_to_string(m_opcate));
                DUMP_TENSOR_TO_TXT(tensor, op_to_string(m_opcate));
            }
        }
    }
    m_complete = true;
}

} // namespace mariana
