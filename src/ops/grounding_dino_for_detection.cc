/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/grounding_dino_for_detection.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-08-27:08:43:55
 * Description:
 * 
 */

#include <models/model_param.h>

#include <utils/mariana_define.h>

#include <ops/matmul.h>
#include <ops/backend/cpu/matmul.h>
#include <ops/backend/cpu/sigmoid.h>
#include <ops/backend/cpu/max.h>
#include <ops/grounding_dino_for_detection.h>
#include <ops/backend/cpu/grounding_dino_utils.h>

namespace mariana {

GroundingDinoForDetectionFunc::~GroundingDinoForDetectionFunc() {
    delete m_bbox_embed_func0;
    delete m_bbox_embed_func1;
    delete m_bbox_embed_func2;
}

void GroundingDinoForDetectionFunc::set_thread_pool(ThreadPool* tp) {
    m_tp = tp;
    m_bbox_embed_func0->set_thread_pool(tp);
    m_bbox_embed_func1->set_thread_pool(tp);
    m_bbox_embed_func2->set_thread_pool(tp);
}

bool GroundingDinoForDetectionFunc::init(const ModelParam& param, const std::string& node_name) {
    m_decoder_layers = param.decoder_layers;
    m_bbox_embed_func0 = new MatMulFunc{};
    m_bbox_embed_func1 = new MatMulFunc{};
    m_bbox_embed_func2 = new MatMulFunc{};
    ModelParam __param = param;
    __param.own_weight = false;
    __param.act_cate = OpCategory::RELU;
    m_bbox_embed_func0->init(__param, "bbox_embed.0.layers.0");
    m_bbox_embed_func1->init(__param, "bbox_embed.0.layers.1");
    __param.act_cate = OpCategory::None;
    m_bbox_embed_func2->init(__param, "bbox_embed.0.layers.2");
    return true;
}

bool GroundingDinoForDetectionFunc::plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (outputs.empty()) { // 1.delta_bbox 2.scores 3.outputs_class
        outputs.resize(3);
    }
    Tensor enc_text_hidden_states = inputs[2*m_decoder_layers];
    for (int32_t i = 0; i < m_decoder_layers; ++i) {
        Tensor hidden_states = inputs[2*i+0];
        Tensor reference = inputs[2*i+1];
        tensor_list _outputs = {m_bbox_embed_out0};
        m_bbox_embed_func0->plan_forward({hidden_states}, _outputs, context);
        _outputs = {m_bbox_embed_out1};
        m_bbox_embed_func1->plan_forward({m_bbox_embed_out0}, _outputs, context);
        _outputs = {outputs[0]};
        m_bbox_embed_func2->plan_forward({m_bbox_embed_out1}, _outputs, context);
        outputs[2].try_realloc({hidden_states.dim_at(0), hidden_states.dim_at(1), enc_text_hidden_states.dim_at(1)}, enc_text_hidden_states.dtype());
        break;
    }
    outputs[1].try_realloc({outputs[2].dim_at(0), outputs[2].dim_at(1)}, outputs[2].dtype());
    return true;
}

bool GroundingDinoForDetectionFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    Tensor enc_text_hidden_states = inputs[2*m_decoder_layers];
    auto _shape = enc_text_hidden_states.dims();
    enc_text_hidden_states.reshape({enc_text_hidden_states.dim_at(0)*enc_text_hidden_states.dim_at(1), enc_text_hidden_states.dim_at(2)});
    for (int32_t i = m_decoder_layers-1; i < m_decoder_layers; ++i) {
        Tensor hidden_states = inputs[2*i+0];
        Tensor reference = inputs[2*i+1];
        tensor_list _outputs = {m_bbox_embed_out0};
        m_bbox_embed_func0->on_forward({hidden_states}, _outputs, context);
        _outputs = {m_bbox_embed_out1};
        m_bbox_embed_func1->on_forward({m_bbox_embed_out0}, _outputs, context);
        _outputs = {outputs[0]};
        m_bbox_embed_func2->on_forward({m_bbox_embed_out1}, _outputs, context);
       
        _parallel_async(m_tp, outputs[0].total_size(), decoder_reference_points_correct, std::ref(outputs[0]), std::ref(reference), std::ref(outputs[0]), 1e-5);
        
        Tensor __bias;
        _parallel_async(m_tp, hidden_states.dim_at(0)*hidden_states.dim_at(1), matmul, std::ref(hidden_states), std::ref(enc_text_hidden_states), std::ref(__bias), std::ref(outputs[2]), 1.f, 1.f, OpCategory::None);
        m_tp->wait_work_complete();
        _parallel_sync(m_tp, outputs[2].total_size(), sigmoid, std::ref(outputs[2]), std::ref(outputs[2]));
        
        Tensor scores = outputs[1];
        _parallel_async(m_tp, outputs[2].dim_at(0)*outputs[2].dim_at(1), max_last_dim_spilt, std::ref(outputs[2]), std::ref(scores));
        int32_t h = 1, w = 1;
        if (context.runtime_info.img_hws.size() == 2) {
            h = context.runtime_info.img_hws[0];
            w = context.runtime_info.img_hws[1];
        }
        
        _parallel_async(m_tp, outputs[0].dim_at(0)*outputs[0].dim_at(1), bbox_center_to_corners, std::ref(outputs[0]), h, w, std::ref(outputs[0]));
        
        m_tp->wait_work_complete();
    }
    enc_text_hidden_states.reshape(_shape);
    return true;
}

} // namespace mariana
