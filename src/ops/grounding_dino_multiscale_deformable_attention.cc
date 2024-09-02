/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/grounding_dino_multiscale_deformable_attention.cc
 * Authors    : lqwang@pandora
 * Create Time: 2024-07-21:08:04:45
 * Description:
 * 
 */

#include <utils/mariana_define.h>

#include <models/model_param.h>

#include <ops/math.h>
#include <ops/backend/cpu/matmul.h>
#include <ops/backend/cpu/softmax.h>
#include <ops/backend/cpu/mhs_attention.h>
#include <ops/grounding_dino_encoder_before.h>
#include <ops/grounding_dino_multiscale_deformable_attention.h>

namespace mariana {

GroundingDinoMultiscaleDeformableAttention::~GroundingDinoMultiscaleDeformableAttention() {
    delete m_add_func;
}

bool GroundingDinoMultiscaleDeformableAttention::init(const ModelParam& param, const std::string& node_name) {
    m_n_levels = param.num_feature_levels;
    m_n_heads  = param.encoder_attention_heads;
    m_n_points = param.encoder_n_points;
    m_d_model  = param.d_model;
    ModelParam::SafeTensorInfo sti;
    // TODO: check weight match the parameter.
    TRY_STL(sti = param.sti_map.at(node_name+".value_proj.weight"), return false);
    Tensor value_proj_weight(sti.shape, DataOn::CPU, sti.data, sti.dtype, true/*move_data*/);
    m_value_proj_weight = value_proj_weight;

    TRY_STL(sti = param.sti_map.at(node_name+".value_proj.bias"), return false);
    Tensor value_proj_bias(sti.shape, DataOn::CPU, sti.data, sti.dtype, true/*move_data*/);
    m_value_proj_bias = value_proj_bias;

    TRY_STL(sti = param.sti_map.at(node_name+".sampling_offsets.weight"), return false);
    Tensor sampling_offsets_weight(sti.shape, DataOn::CPU, sti.data, sti.dtype, true/*move_data*/);
    m_sampling_offsets_weight = sampling_offsets_weight;

    TRY_STL(sti = param.sti_map.at(node_name+".sampling_offsets.bias"), return false);
    Tensor sampling_offsets_bias(sti.shape, DataOn::CPU, sti.data, sti.dtype, true/*move_data*/);
    m_sampling_offsets_bias = sampling_offsets_bias;

    TRY_STL(sti = param.sti_map.at(node_name+".attention_weights.weight"), return false);
    Tensor attention_weights_weight(sti.shape, DataOn::CPU, sti.data, sti.dtype, true/*move_data*/);
    m_attention_weights_weight = attention_weights_weight;

    TRY_STL(sti = param.sti_map.at(node_name+".attention_weights.bias"), return false);
    Tensor attention_weights_bias(sti.shape, DataOn::CPU, sti.data, sti.dtype, true/*move_data*/);
    m_attention_weights_bias = attention_weights_bias;

    TRY_STL(sti = param.sti_map.at(node_name+".output_proj.weight"), return false);
    Tensor output_proj_weight(sti.shape, DataOn::CPU, sti.data, sti.dtype, true/*move_data*/);
    m_output_proj_weight = output_proj_weight;

    TRY_STL(sti = param.sti_map.at(node_name+".output_proj.bias"), return false);
    Tensor output_proj_bias(sti.shape, DataOn::CPU, sti.data, sti.dtype, true/*move_data*/);
    m_output_proj_bias = output_proj_bias;

    m_add_func = new AddFunc{};
    
    return true;
}

void GroundingDinoMultiscaleDeformableAttention::set_thread_pool(ThreadPool* tp) {
    m_tp = tp;
    m_add_func->set_thread_pool(tp);
}

bool GroundingDinoMultiscaleDeformableAttention::plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    Tensor hidden_states = inputs[0];
    Tensor encoder_hidden_states = inputs[1];
    Tensor position_embeddings = inputs[2];
    tensor_list _inputs = {hidden_states, position_embeddings};
    tensor_list _outputs = {m_add_out};
    m_add_func->plan_forward(_inputs, _outputs, context);
    
    int32_t vnb = encoder_hidden_states.dim_at(0);
    int32_t vnr = encoder_hidden_states.dim_at(1);
    int32_t vnc = m_value_proj_weight.dim_at(0);
    m_value_out.try_realloc({vnb, vnr, vnc}, hidden_states.dtype());

    vnb = hidden_states.dim_at(0);
    vnr = hidden_states.dim_at(1);
    vnc = m_sampling_offsets_weight.dim_at(0);
    m_sampling_offsets_out.try_realloc({vnb, vnr, vnc}, hidden_states.dtype());
    m_msda_out.try_realloc({vnb, vnr, vnc}, hidden_states.dtype());
    if (outputs.empty()) {
        Tensor output_proj;
        output_proj.try_realloc({vnb, vnr, vnc}, hidden_states.dtype());
        outputs.push_back(output_proj);
    } else {
        outputs[0].try_realloc({vnb, vnr, vnc}, hidden_states.dtype());
    }
    
    vnc = m_attention_weights_weight.dim_at(0);
    m_attention_weights_out.try_realloc({vnb, vnr, vnc}, hidden_states.dtype());
    m_attention_weights_softmax_out.try_realloc({vnb, vnr, m_n_heads, m_n_levels*m_n_points}, m_attention_weights_out.dtype());
        
    return true;
}

bool GroundingDinoMultiscaleDeformableAttention::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    Tensor hidden_states = inputs[0];
    Tensor encoder_hidden_states = inputs[1];
    Tensor position_embeddings = inputs[2];
    tensor_list _inputs = {hidden_states, position_embeddings};
    tensor_list _outputs = {m_add_out};
    m_add_func->on_forward(_inputs, _outputs, context);
    _parallel_async(m_tp, encoder_hidden_states.dim_at(0)*encoder_hidden_states.dim_at(1), matmul, std::ref(encoder_hidden_states), std::ref(m_value_proj_weight), std::ref(m_value_proj_bias), std::ref(m_value_out), 1.f, 1.f, OpCategory::None);
    _parallel_async(m_tp, hidden_states.dim_at(0)*hidden_states.dim_at(1), matmul, std::ref(m_add_out), std::ref(m_sampling_offsets_weight), std::ref(m_sampling_offsets_bias), std::ref(m_sampling_offsets_out), 1.f, 1.f, OpCategory::None);
    _parallel_async(m_tp, hidden_states.dim_at(0)*hidden_states.dim_at(1), matmul, std::ref(m_add_out), std::ref(m_attention_weights_weight), std::ref(m_attention_weights_bias), std::ref(m_attention_weights_out), 1.f, 1.f, OpCategory::None);
    m_tp->wait_work_complete();
    m_attention_weights_out.reshape({m_attention_weights_out.dim_at(0), m_attention_weights_out.dim_at(1), m_n_heads, m_n_levels*m_n_points});
    _parallel_sync(m_tp, m_attention_weights_out.dim_at(0)*m_attention_weights_out.dim_at(1)*m_attention_weights_out.dim_at(2), softmax, std::ref(m_attention_weights_out),std::ref(m_attention_weights_softmax_out));
    m_attention_weights_softmax_out.reshape({m_attention_weights_softmax_out.dim_at(0), m_attention_weights_softmax_out.dim_at(1), m_n_heads, m_n_levels, m_n_points});
    
    Tensor reference_point = inputs[3];
    auto _shape = reference_point.dims();
    m_sampling_offsets_out.reshape({m_sampling_offsets_out.dim_at(0), m_sampling_offsets_out.dim_at(1), m_n_heads, m_n_levels, m_n_points, 2});
    if (reference_point.dim_size() == 4) { // 1 20906 4 2
        reference_point.reshape({reference_point.dim_at(0), reference_point.dim_at(1), 1, m_n_levels, 1, 2});
    } else if (reference_point.dim_size() == 3) { // 1 900 4
        reference_point.reshape({reference_point.dim_at(0), reference_point.dim_at(1), 1, 1, 1, 4});
    } else {
        MLOG(ERROR)<<"Unsupport reference points dims:"<<reference_point.dim_size();
        return false;
    }
    
    m_value_out.reshape({m_value_out.dim_at(0), m_value_out.dim_at(1), m_n_heads, m_d_model/m_n_heads});
    GroundingDinoEncoderBeforeFunc::SpatialShapes* sp_shape = static_cast<GroundingDinoEncoderBeforeFunc::SpatialShapes*>(context.runtime_info.anything);
    _parallel_sync(m_tp, m_msda_out.dim_at(0)*m_msda_out.dim_at(1), multi_scale_deformable_attention, std::ref(*sp_shape), std::ref(m_value_out), std::ref(m_attention_weights_softmax_out), std::ref(m_sampling_offsets_out), std::ref(reference_point), std::ref(m_msda_out));
    reference_point.reshape(_shape);
    
    _parallel_sync(m_tp, m_msda_out.dim_at(0)*m_msda_out.dim_at(1), matmul, std::ref(m_msda_out), std::ref(m_output_proj_weight), std::ref(m_output_proj_bias), std::ref(outputs[0]), 1.f, 1.f, OpCategory::None);
    return true;
}

} // namespace mariana 
