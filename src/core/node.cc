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

void Node::set_thread_pool(ThreadPool* tp) {
    m_op->set_thread_pool(tp);
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

bool Node::plan_forward(const tensor_list& inputs, ExeContext& context) {
    MLOG_IF(ERROR, !m_op)<<"m_op null";
    m_op->plan_forward(inputs, m_otensors, context);
    return true;
}

void Node::forward(const tensor_list& inputs, ExeContext& context) {
    plan_forward(inputs, context);
    {
        AUTOTIME(op_to_string(m_opcate).c_str());
        bool ok = m_op->on_forward(inputs, m_otensors, context);
        MLOG_IF(ERROR, !ok)<<"Node:"<<op_to_string(m_opcate)<<" forward failed";
    }
    
    if (get_env("MAR_LAYER_RES_DUMP") == "1") {
        for (auto& tensor : m_otensors) {
            DUMP_TENSOR_TO_BIN(tensor, op_to_string(m_opcate));
            DUMP_TENSOR_TO_TXT(tensor, op_to_string(m_opcate));
        }
    }
}

} // namespace mariana
