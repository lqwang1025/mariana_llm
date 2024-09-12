/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/grounding_dino_bi_mhs_attention.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-11:09:28:54
 * Description:
 * 
 */

#include <cmath>
#include <cfloat>

#include <absl/strings/str_replace.h>
#include <models/model_param.h>
#include <utils/mariana_define.h>
#include <core/tensor_utils.h>

#include <ops/math.h>
#include <ops/backend/cpu/matmul.h>
#include <ops/backend/cpu/permute.h>
#include <ops/grounding_dino_bi_mhs_attention.h>
#include <ops/backend/cpu/mhs_attention.h>

namespace mariana {

bool GroundingDinoBiMHSAttentionFunc::init(const ModelParam& param, const std::string& node_name) {
    TRACE();
    if (param.n_embd % param.n_head != 0) {
        MLOG(ERROR)<<"The hidden size "<<param.n_embd
                   <<" is not a multiple of the number of attention "<<"heads "<<param.n_head;
        return false;
    }
    m_attention_head_size = param.n_embd/param.n_head;
    m_n_head = param.n_head;
    m_scale = 1/sqrt(m_attention_head_size);
    ModelParam::SafeTensorInfo sti;
    // 1. vision_proj
    TRY_STL(sti = param.sti_map.at(node_name+".vision_proj.weight"), return false);
    Tensor vision_proj(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    m_vision_proj = vision_proj.deepcopy();
    TRY_STL(sti = param.sti_map.at(node_name+".vision_proj.bias"), return false);
    Tensor vision_proj_bias(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    m_vision_proj_bias = vision_proj_bias.deepcopy();
    // 2. text_proj
    TRY_STL(sti = param.sti_map.at(node_name+".text_proj.weight"), return false);
    Tensor text_proj(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    m_text_proj = text_proj.deepcopy();
    TRY_STL(sti = param.sti_map.at(node_name+".text_proj.bias"), return false);
    Tensor text_proj_bias(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    m_text_proj_bias = text_proj_bias.deepcopy();
    // 3. vision_value
    TRY_STL(sti = param.sti_map.at(node_name+".values_vision_proj.weight"), return false);
    Tensor vision_value(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    m_vision_value_proj = vision_value.deepcopy();
    TRY_STL(sti = param.sti_map.at(node_name+".values_vision_proj.bias"), return false);
    Tensor vision_value_bias(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    m_vision_value_proj_bias = vision_value_bias.deepcopy();
    // 4. text_value
    TRY_STL(sti = param.sti_map.at(node_name+".values_text_proj.weight"), return false);
    Tensor text_value(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    m_text_value_proj = text_value.deepcopy();
    TRY_STL(sti = param.sti_map.at(node_name+".values_text_proj.bias"), return false);
    Tensor text_value_bias(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    m_text_value_proj_bias = text_value_bias.deepcopy();
    // 5. out_vision
    TRY_STL(sti = param.sti_map.at(node_name+".out_vision_proj.weight"), return false);
    Tensor out_vision(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    m_out_vision_proj = out_vision.deepcopy();
    TRY_STL(sti = param.sti_map.at(node_name+".out_vision_proj.bias"), return false);
    Tensor out_vision_bias(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    m_out_vision_proj_bias = out_vision_bias.deepcopy();
    
    std::string _name = absl::StrReplaceAll(node_name, {{".attn", ""}});
    TRY_STL(sti = param.sti_map.at(_name+".vision_param"), return false);
    Tensor vision_param(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    
    // 6. out_text
    TRY_STL(sti = param.sti_map.at(node_name+".out_text_proj.weight"), return false);
    Tensor out_text(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    m_out_text_proj = out_text.deepcopy();
    TRY_STL(sti = param.sti_map.at(node_name+".out_text_proj.bias"), return false);
    Tensor out_text_bias(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    m_out_text_proj_bias = out_text_bias.deepcopy();
    
    TRY_STL(sti = param.sti_map.at(_name+".text_param"), return false);
    Tensor text_param(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    
    ThreadPool* tp = new ThreadPool(ThreadPool::default_num_threads());
    MulFunc mul_func;
    mul_func.set_thread_pool(tp);
    tensor_list inputs = {out_vision, vision_param};
    tensor_list output = {m_out_vision_proj};
    ExeContext context;
    mul_func.on_forward(inputs, output, context);
    inputs = {out_vision_bias, vision_param};
    output = {m_out_vision_proj_bias};
    mul_func.on_forward(inputs, output, context);
    inputs = {out_text, text_param};
    output = {m_out_text_proj};
    mul_func.on_forward(inputs, output, context);
    inputs = {out_text_bias, text_param};
    output = {m_out_text_proj_bias};
    mul_func.on_forward(inputs, output, context);
    delete tp;
    return true;
}

bool GroundingDinoBiMHSAttentionFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    const Tensor& vision_features = inputs[0];
    int32_t vnb = vision_features.dim_at(0);
    int32_t vnr = vision_features.dim_at(1);
    int32_t vnc = m_vision_proj.dim_at(0);
    m_vision_query_states.try_realloc({vnb, vnr, vnc}, vision_features.dtype());
    m_vision_value_states.try_realloc({vnb, vnr, vnc}, vision_features.dtype());
    m_vision_value_states_perm.try_realloc({1, m_n_head, m_attention_head_size, vnr}, vision_features.dtype());
    
    const Tensor& text_features = inputs[1];
    int32_t tnb = text_features.dim_at(0);
    int32_t tnr = text_features.dim_at(1);
    int32_t tnc = m_text_proj.dim_at(0);
    m_text_key_states.try_realloc({tnb, tnr, tnc}, vision_features.dtype());
    m_text_value_states.try_realloc({tnb, tnr, tnc}, text_features.dtype());
    m_text_value_states_perm.try_realloc({1, m_n_head, m_attention_head_size, tnr}, text_features.dtype());

    m_vision_output.try_realloc({1, vnr, m_attention_head_size*m_n_head}, vision_features.dtype());
    m_text_output.try_realloc({1, tnr, m_attention_head_size*m_n_head}, vision_features.dtype());

    if (outputs.empty()) {
        Tensor attn_weight;
        attn_weight.try_realloc({m_n_head, vnr, tnr}, vision_features.dtype()); // 1
        Tensor text_attn_weight;
        text_attn_weight.try_realloc({m_n_head, tnr, vnr}, vision_features.dtype()); // 2
        Tensor out_vision_proj_out;
        out_vision_proj_out.try_realloc({1, vnr, m_attention_head_size}, vision_features.dtype());//3
        Tensor out_text_proj_out;
        out_text_proj_out.try_realloc({1, tnr, m_attention_head_size}, text_features.dtype());//4
        outputs = {out_vision_proj_out, attn_weight, out_text_proj_out, text_attn_weight};
    } else {
        outputs[0].try_realloc({1, vnr, m_attention_head_size}, vision_features.dtype());
        outputs[1].try_realloc({m_n_head, vnr, tnr}, vision_features.dtype()); // 1
        outputs[2].try_realloc({1, tnr, m_attention_head_size}, text_features.dtype());//4
        outputs[3].try_realloc({m_n_head, tnr, vnr}, vision_features.dtype()); // 2
    }
    return true;
}

bool GroundingDinoBiMHSAttentionFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {

    _parallel_async(m_tp, inputs[0].dim_at(0)*inputs[0].dim_at(1), matmul, std::ref(inputs[0]),
                    std::ref(m_vision_proj), std::ref(m_vision_proj_bias), std::ref(m_vision_query_states), m_scale, m_scale, OpCategory::None);
    _parallel_async(m_tp, inputs[1].dim_at(0)*inputs[1].dim_at(1), matmul, std::ref(inputs[1]),
                    std::ref(m_text_proj), std::ref(m_text_proj_bias), std::ref(m_text_key_states), 1.f, 1.f, OpCategory::None);
    _parallel_async(m_tp, inputs[0].dim_at(0)*inputs[0].dim_at(1), matmul, std::ref(inputs[0]),
                    std::ref(m_vision_value_proj), std::ref(m_vision_value_proj_bias), std::ref(m_vision_value_states), 1.f, 1.f, OpCategory::None);
    _parallel_async(m_tp, inputs[1].dim_at(0)*inputs[1].dim_at(1), matmul, std::ref(inputs[1]),
                    std::ref(m_text_value_proj), std::ref(m_text_value_proj_bias), std::ref(m_text_value_states), 1.f, 1.f, OpCategory::None);
    m_tp->wait_work_complete();
    uint8_t perms[4] = {0, 2, 3, 1};
    m_vision_value_states.reshape({1, m_vision_value_states.dim_at(1), m_n_head, m_attention_head_size});
    _parallel_async(m_tp, m_vision_value_states.total_size(), permute4, std::ref(m_vision_value_states), std::ref(m_vision_value_states_perm), perms);
    m_text_value_states.reshape({1, m_text_value_states.dim_at(1), m_n_head, m_attention_head_size});
    _parallel_async(m_tp, m_text_value_states.total_size(), permute4, std::ref(m_text_value_states), std::ref(m_text_value_states_perm), perms);
    m_tp->wait_work_complete();

    std::atomic<float> max_val = -FLT_MAX;
    _parallel_sync(m_tp, m_vision_query_states.dim_at(0)*m_vision_query_states.dim_at(1), bimhs_grdino_qk_dot_batch_split, std::ref(m_vision_query_states), std::ref(m_text_key_states), std::ref(outputs[1]), m_n_head, m_attention_head_size, std::ref(max_val));
    
    _parallel_sync(m_tp, outputs[1].dim_at(0), bimhs_grdino_attn_clamp_bt_split, std::ref(outputs[1]), std::ref(outputs[3]), max_val.load(), -50000, 50000);
    
    _parallel_async(m_tp, outputs[1].dim_at(0)*outputs[1].dim_at(1), bimhs_grdino_attn_v_dot_batch_split, std::ref(outputs[1]), std::ref(m_text_value_states_perm), std::ref(m_vision_output), m_n_head, m_attention_head_size);
    _parallel_async(m_tp, outputs[3].dim_at(0)*outputs[3].dim_at(1), bimhs_grdino_attn_v_dot_batch_split, std::ref(outputs[3]), std::ref(m_vision_value_states_perm), std::ref(m_text_output), m_n_head, m_attention_head_size);
    m_tp->wait_work_complete();
    
    _parallel_async(m_tp, m_vision_output.dim_at(0)*m_vision_output.dim_at(1), matmul, std::ref(m_vision_output),
                    std::ref(m_out_vision_proj), std::ref(m_out_vision_proj_bias), std::ref(outputs[0]), 1.f, 1.f, OpCategory::None);
    _parallel_async(m_tp, m_text_output.dim_at(0)*m_text_output.dim_at(1), matmul, std::ref(m_text_output),
                    std::ref(m_out_text_proj), std::ref(m_out_text_proj_bias), std::ref(outputs[2]), 1.f, 1.f, OpCategory::None);
    m_tp->wait_work_complete();
    return true;
}

} // namespace mariana
