/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : self_attention.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-09-26:16:59:31
 * Description:
 * 
 */

#include <core/tensor_utils.h>

#include <ops/self_attention.h>
#include <ops/backend/gpu/impl/matmul.h>
#include <ops/backend/gpu/impl/mhs_attention.h>

#include <core/node.h>
#include <core/backend/gpu/cuda_common.h>

namespace mariana {

bool SelfAttentionFunc::plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    cuda_set_device(cuda_ctx->device);
    if (m_q_weight.device() != DataOn::GPU) {
        m_q_weight = m_q_weight.cuda(cuda_ctx->stream());
    }
    if (m_q_bias.device() != DataOn::GPU) {
        m_q_bias = m_q_bias.cuda(cuda_ctx->stream());
    }
    if (m_k_weight.device() != DataOn::GPU) {
        m_k_weight = m_k_weight.cuda(cuda_ctx->stream());
    }
    if (m_k_bias.device() != DataOn::GPU) {
        m_k_bias = m_k_bias.cuda(cuda_ctx->stream());
    }
    if (m_v_weight.device() != DataOn::GPU) {
        m_v_weight = m_v_weight.cuda(cuda_ctx->stream());
    }
    if (m_v_bias.device() != DataOn::GPU) {
        m_v_bias = m_v_bias.cuda(cuda_ctx->stream());
    }
    if (outputs.empty()) {
        outputs.push_back(Tensor(DataOn::GPU));
    }
    outputs[0].try_realloc(inputs[0].dims(), inputs[0].dtype());
    if (m_q_o.device() != DataOn::GPU) {
        m_q_o = Tensor(DataOn::GPU);
    }
    if (m_k_o.device() != DataOn::GPU) {
        m_k_o = Tensor(DataOn::GPU);
    }
    if (m_v_o.device() != DataOn::GPU) {
        m_v_o = Tensor(DataOn::GPU);
    }
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

bool SelfAttentionFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
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
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    _parallel_async(m_tp, query.dim_at(0), matmul, std::ref(query), std::ref(m_q_weight), std::ref(m_q_bias), std::ref(m_q_o), 1.f, 1.f, OpCategory::None, cuda_ctx);
    _parallel_async(m_tp, key.dim_at(0), matmul, std::ref(key), std::ref(m_k_weight), std::ref(m_k_bias), std::ref(m_k_o), 1.f, 1.f, OpCategory::None, cuda_ctx);
    _parallel_async(m_tp, value.dim_at(0), matmul, std::ref(value), std::ref(m_v_weight), std::ref(m_v_bias), std::ref(m_v_o), 1.f, 1.f, OpCategory::None, cuda_ctx);
    m_tp->wait_work_complete();
    if (pos_mask.total_size() == 0) {
        if (attn_mask.total_size() == 0) {
            _parallel_sync(m_tp, m_q_o.dim_at(0)*m_q_o.dim_at(1), mhs_attention_batch_split, std::ref(m_q_o),
                           std::ref(m_k_o), std::ref(m_v_o), std::ref(outputs[0]), m_n_head, m_attention_head_size);
        } else {
            _parallel_sync(m_tp, m_q_o.dim_at(0), mhs_mask_attention, std::ref(m_q_o), std::ref(m_k_o), std::ref(m_v_o), std::ref(attn_mask), std::ref(outputs[0]), m_n_head, m_attention_head_size, cuda_ctx);
        }
    } else {
        MLOG(INFO)<<"TODO";
    }
    cuda_ctx->stream_sync(cuda_ctx->stream(0));
    Tensor q = m_q_o.cpu();
    DUMP_TENSOR_TO_TXT(q, "q_o");
    Tensor k = m_k_o.cpu();
    DUMP_TENSOR_TO_TXT(k, "k_o");
    Tensor v = m_v_o.cpu();
    DUMP_TENSOR_TO_TXT(v, "v_o");
    Tensor out = outputs[0].cpu();
    DUMP_TENSOR_TO_TXT(out, "out");
    exit(0);
    return true;
}
    
} // namespace mariana
