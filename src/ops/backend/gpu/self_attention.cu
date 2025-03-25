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

#include <cmath>

#include <ops/self_attention.h>
#include <ops/backend/gpu/impl/matmul.h>
#include <ops/backend/gpu/impl/permute.h>
#include <ops/backend/gpu/impl/tile.h>
#include <ops/backend/gpu/impl/softmax.h>
#include <ops/backend/gpu/impl/mhs_attention.h>
#include <ops/backend/gpu/impl/math.h>

#include <core/node.h>
#include <core/backend/gpu/cuda_common.h>
#include <core/tensor_utils.h>

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
    if (m_o_weight.device() != DataOn::GPU) {
        m_o_weight = m_o_weight.cuda(cuda_ctx->stream());
    }
    if (m_o_bias.device() != DataOn::GPU) {
        m_o_bias = m_o_bias.cuda(cuda_ctx->stream());
    }
    if (outputs.empty()) {
        outputs.push_back(Tensor(DataOn::GPU));
    }
    outputs[0].try_realloc(inputs[0].dims(), inputs[0].dtype());
    if (m_q_o.device() != DataOn::GPU) {
        m_q_o = Tensor(DataOn::GPU);
        m_qtrans_o = Tensor(DataOn::GPU);
    }
    if (m_k_o.device() != DataOn::GPU) {
        m_k_o = Tensor(DataOn::GPU);
        m_ktrans_o = Tensor(DataOn::GPU);
    }
    if (m_v_o.device() != DataOn::GPU) {
        m_v_o = Tensor(DataOn::GPU);
        m_vtrans_o = Tensor(DataOn::GPU);
    }
    int32_t nb = inputs[0].dim_at(0);
    int32_t nr = inputs[0].dim_at(1);
    int32_t nc = m_q_weight.dim_at(0);
    m_q_o.try_realloc({nb, nr, nc}, inputs[0].dtype());
    m_qtrans_o.try_realloc({nb, nc/m_attention_head_size, nr, m_attention_head_size}, inputs[0].dtype());
    
    nc = m_k_weight.dim_at(0);
    m_k_o.try_realloc({nb, nr, nc}, inputs[0].dtype());
    m_ktrans_o.try_realloc({nb, nc/m_attention_head_size, nr, m_attention_head_size}, inputs[0].dtype());
    
    nc = m_v_weight.dim_at(0);
    m_v_o.try_realloc({nb, nr, nc}, inputs[0].dtype());
    m_vtrans_o.try_realloc({nb, nc/m_attention_head_size, nr, m_attention_head_size}, inputs[0].dtype());
    return true;
}

bool SelfAttentionFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    // inputs squence is : hidden_states, sin, cos
    Tensor hidden_states = inputs[0];
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    _parallel_async(m_tp, hidden_states.dim_at(0), matmul, std::ref(hidden_states), std::ref(m_q_weight), std::ref(m_q_bias), std::ref(m_q_o), 1.f, 1.f, OpCategory::None, cuda_ctx);
    _parallel_async(m_tp, hidden_states.dim_at(0), matmul, std::ref(hidden_states), std::ref(m_k_weight), std::ref(m_k_bias), std::ref(m_k_o), 1.f, 1.f, OpCategory::None, cuda_ctx);
    _parallel_async(m_tp, hidden_states.dim_at(0), matmul, std::ref(hidden_states), std::ref(m_v_weight), std::ref(m_v_bias), std::ref(m_v_o), 1.f, 1.f, OpCategory::None, cuda_ctx);
    m_tp->wait_work_complete();
    
    m_q_o.reshape({hidden_states.dim_at(0), hidden_states.dim_at(1),
            m_q_o.dim_at(2)/m_attention_head_size, m_attention_head_size});
    uint8_t perms[4] = {0, 2, 1, 3};
    _parallel_async(m_tp, 1, permute4, std::ref(m_q_o), std::ref(m_qtrans_o), perms, cuda_ctx);
    m_k_o.reshape({hidden_states.dim_at(0), hidden_states.dim_at(1),
            m_k_o.dim_at(2)/m_attention_head_size, m_attention_head_size});
    _parallel_async(m_tp, 1, permute4, std::ref(m_k_o), std::ref(m_ktrans_o), perms, cuda_ctx);
    m_v_o.reshape({hidden_states.dim_at(0), hidden_states.dim_at(1),
            m_v_o.dim_at(2)/m_attention_head_size, m_attention_head_size});
    _parallel_async(m_tp, 1, permute4, std::ref(m_v_o), std::ref(m_vtrans_o), perms, cuda_ctx);
    m_tp->wait_work_complete();
    
    Tensor sin = inputs[1];
    Tensor cos = inputs[2];
    m_q_o.reshape(m_qtrans_o.dims());
    _parallel_async(m_tp, m_qtrans_o.dim_at(0), apply_rotary_pos_emb, std::ref(m_qtrans_o), std::ref(sin), std::ref(cos), std::ref(m_q_o), cuda_ctx);
    m_k_o.reshape(m_ktrans_o.dims());
    _parallel_async(m_tp, m_ktrans_o.dim_at(0), apply_rotary_pos_emb, std::ref(m_ktrans_o), std::ref(sin), std::ref(cos), std::ref(m_k_o), cuda_ctx);
    m_tp->wait_work_complete();

// repeat key value
    uint32_t num_key_value_groups = m_qtrans_o.dim_at(1)/m_vtrans_o.dim_at(1);
    m_k_o.reshape({m_k_o.dim_at(0)*m_k_o.dim_at(1), 1, m_k_o.dim_at(2), m_k_o.dim_at(3)});
    m_qtrans_o.try_realloc({m_k_o.dim_at(0), (int32_t)num_key_value_groups, m_k_o.dim_at(2), m_k_o.dim_at(3)}, m_qtrans_o.dtype());
    uint32_t repeats[4] = {1, num_key_value_groups, 1, 1};
    _parallel_async(m_tp, 1, tile4, std::ref(m_k_o), std::ref(m_qtrans_o), repeats, cuda_ctx);
        
    m_vtrans_o.reshape({m_vtrans_o.dim_at(0)*m_vtrans_o.dim_at(1), 1, m_vtrans_o.dim_at(2), m_vtrans_o.dim_at(3)});
    m_ktrans_o.try_realloc({m_vtrans_o.dim_at(0), (int32_t)num_key_value_groups, m_vtrans_o.dim_at(2), m_vtrans_o.dim_at(3)}, m_vtrans_o.dtype());
    _parallel_async(m_tp, 1, tile4, std::ref(m_vtrans_o), std::ref(m_ktrans_o), repeats, cuda_ctx);

    m_tp->wait_work_complete();
    Tensor query = m_q_o;
    m_qtrans_o.reshape(m_q_o.dims());
    Tensor key = m_qtrans_o;
    m_ktrans_o.reshape(m_q_o.dims());
    Tensor value = m_ktrans_o;
    
    Tensor attn_weights = m_k_o;
    Tensor place_holder;
    attn_weights.try_realloc({query.dim_at(0)*query.dim_at(1), query.dim_at(2), key.dim_at(2)}, attn_weights.dtype());
    query.reshape({query.dim_at(0)*query.dim_at(1), query.dim_at(2), query.dim_at(3)});
    key.reshape({key.dim_at(0)*key.dim_at(1), key.dim_at(2), key.dim_at(3)});
    float scale = 1/sqrt(query.dim_at(2));
    _parallel_sync(m_tp, query.dim_at(0), batch_matmul, std::ref(query), std::ref(key), std::ref(place_holder), std::ref(attn_weights), scale, 1.f, OpCategory::None, cuda_ctx);

    Tensor att_mask = inputs[3];
    _parallel_sync(m_tp, attn_weights.total_size(), add_ele, std::ref(attn_weights), std::ref(att_mask), std::ref(attn_weights), cuda_ctx);
    
    Tensor softmaxed = m_q_o;
    softmaxed.try_realloc(attn_weights.dims(), attn_weights.dtype());
    _parallel_sync(m_tp, attn_weights.dim_at(0), softmax3, std::ref(attn_weights), std::ref(softmaxed), -1/*dim*/, cuda_ctx);
    
    attn_weights.try_realloc({value.dim_at(0)*value.dim_at(1), softmaxed.dim_at(2), value.dim_at(3)}, attn_weights.dtype());
    
    perms[1] = 1;
    perms[2] = 3;
    perms[3] = 2;
    m_vtrans_o.try_realloc({value.dim_at(0), value.dim_at(1), value.dim_at(3), value.dim_at(2)}, value.dtype());
    _parallel_sync(m_tp, 1, permute4, std::ref(value), std::ref(m_vtrans_o), perms, cuda_ctx);
    value = m_vtrans_o;
    value.reshape({value.dim_at(0)*value.dim_at(1), value.dim_at(2), value.dim_at(3)});
    _parallel_sync(m_tp, softmaxed.dim_at(0), batch_matmul, std::ref(softmaxed), std::ref(value), std::ref(place_holder), std::ref(attn_weights), 1.f, 1.f, OpCategory::None, cuda_ctx);
    
    DUMP_TENSOR_TO_BIN(attn_weights.cpu(), "attn_weights");
    DUMP_TENSOR_TO_TXT(attn_weights.cpu(), "attn_weights");
    DUMP_TENSOR_TO_BIN(softmaxed.cpu(), "softmaxed");
    DUMP_TENSOR_TO_TXT(softmaxed.cpu(), "softmaxed");
    DUMP_TENSOR_TO_BIN(value.cpu(), "value");
    DUMP_TENSOR_TO_TXT(value.cpu(), "value");

    MLOG(INFO)<<inputs.size();
    return true;
}
    
} // namespace mariana
