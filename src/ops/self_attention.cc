/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/self_attention.cc
 * Authors    : lqwang@pandora
 * Create Time: 2024-06-23:15:48:09
 * Description:
 * 
 */

#include <utils/mariana_define.h>
#include <models/model_param.h>

#include <ops/self_attention.h>
#include <ops/backend/cpu/matmul.h>
#include <ops/backend/cpu/mhs_attention.h>

namespace mariana {

bool SelfAttentionFunc::init(const ModelParam& param, const std::string& node_name) {
    TRACE();
    if (param.n_embd % param.n_head != 0) {
        MLOG(ERROR)<<"The hidden size "<<param.n_embd<<" is not a multiple of the number of attention "
                   <<"heads "<<param.n_head;
        return false;
    }

    m_attention_head_size = param.n_embd/param.n_head;
    m_n_head = param.n_head;
    ModelParam::SafeTensorInfo sti;
    TRY_STL(sti = param.sti_map.at(node_name+".query.weight"), return false);
    Tensor q_weight(sti.shape, DataOn::CPU, sti.data, sti.dtype, true/*move_data*/);
    TRY_STL(sti = param.sti_map.at(node_name+".query.bias"), return false);
    Tensor q_bias(sti.shape, DataOn::CPU, sti.data, sti.dtype, true/*move_data*/);
    
    TRY_STL(sti = param.sti_map.at(node_name+".key.weight"), return false);
    Tensor k_weight(sti.shape, DataOn::CPU, sti.data, sti.dtype, true/*move_data*/);
    TRY_STL(sti = param.sti_map.at(node_name+".key.bias"), return false);
    Tensor k_bias(sti.shape, DataOn::CPU, sti.data, sti.dtype, true/*move_data*/);
    
    TRY_STL(sti = param.sti_map.at(node_name+".value.weight"), return false);
    Tensor v_weight(sti.shape, DataOn::CPU, sti.data, sti.dtype, true/*move_data*/);
    TRY_STL(sti = param.sti_map.at(node_name+".value.bias"), return false);
    Tensor v_bias(sti.shape, DataOn::CPU, sti.data, sti.dtype, true/*move_data*/);
    m_q_weight = q_weight;
    m_q_bias   = q_bias;
    m_k_weight = k_weight;
    m_k_bias   = k_bias;
    m_v_weight = v_weight;
    m_v_bias   = v_bias;
    
    return true;
}

bool SelfAttentionFunc::plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (outputs.empty()) {
        outputs.push_back(Tensor(inputs[0].device()));
    }
    outputs[0].try_realloc(inputs[0].dims(), inputs[0].dtype());
    if (inputs.size() == 4) {
        m_q_o.try_realloc(inputs[0].dims(), inputs[0].dtype());
        m_k_o.try_realloc(inputs[1].dims(), inputs[0].dtype());
        m_v_o.try_realloc(inputs[2].dims(), inputs[0].dtype());
    } else {
        m_q_o.try_realloc(inputs[0].dims(), inputs[0].dtype());
        m_k_o.try_realloc(inputs[0].dims(), inputs[0].dtype());
        m_v_o.try_realloc(inputs[0].dims(), inputs[0].dtype());
    }
    return true;
}

bool SelfAttentionFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    Tensor query, key, value, attn_mask, pos_mask;
    if (inputs.size() == 2) {
        query     = inputs[0];
        key       = inputs[0];
        value     = inputs[0];
        attn_mask = inputs[1];
    } else if (inputs.size() == 3) {
        query     = inputs[0];
        key       = inputs[0];
        value     = inputs[0];
        attn_mask = inputs[1];
        pos_mask  = inputs[2];
    } else if (inputs.size() == 4) {
        query     = inputs[0];
        key       = inputs[1];
        value     = inputs[2];
        attn_mask = inputs[3];
    } else {
        MLOG(ERROR)<<"Unsupport input size:"<<inputs.size();
        return false;
    }
    _parallel_async(m_tp, query.dim_at(0)*query.dim_at(1), matmul, std::ref(query),
                    std::ref(m_q_weight), std::ref(m_q_bias), std::ref(m_q_o), 1.f, 1.f, OpCategory::None);
    
    _parallel_async(m_tp, key.dim_at(0)*key.dim_at(1), matmul, std::ref(key),
                    std::ref(m_k_weight), std::ref(m_k_bias), std::ref(m_k_o), 1.f, 1.f, OpCategory::None);
    _parallel_async(m_tp, value.dim_at(0)*value.dim_at(1), matmul, std::ref(value),
                    std::ref(m_v_weight), std::ref(m_v_bias), std::ref(m_v_o), 1.f, 1.f, OpCategory::None);
    m_tp->wait_work_complete();
    // _parallel_sync(m_tp, m_n_head, mhs_mask_attention_head_split, std::ref(m_q_o),
    //                std::ref(m_k_o), std::ref(m_v_o), std::ref(inputs[1]), std::ref(outputs[0]),
    //                m_n_head, m_attention_head_size);
    if (pos_mask.total_size() == 0) {
        if (attn_mask.total_size() == 0) {
            _parallel_sync(m_tp, m_q_o.dim_at(0)*m_q_o.dim_at(1), mhs_attention_batch_split, std::ref(m_q_o),
                           std::ref(m_k_o), std::ref(m_v_o), std::ref(outputs[0]), m_n_head, m_attention_head_size);
        } else {
            _parallel_sync(m_tp, m_q_o.dim_at(0)*m_q_o.dim_at(1), mhs_mask_attention_batch_split, std::ref(m_q_o),
                           std::ref(m_k_o), std::ref(m_v_o), std::ref(attn_mask), std::ref(outputs[0]),
                           m_n_head, m_attention_head_size);
        }
    } else {
        _parallel_sync(m_tp, m_q_o.dim_at(0)*m_q_o.dim_at(1), mhs_swin_mask_attention_batch_split, std::ref(m_q_o),
                       std::ref(m_k_o), std::ref(m_v_o), std::ref(attn_mask), std::ref(pos_mask), std::ref(outputs[0]),
                       m_n_head, m_attention_head_size);
    }
    
    return true;
}

} // namespace mariana
