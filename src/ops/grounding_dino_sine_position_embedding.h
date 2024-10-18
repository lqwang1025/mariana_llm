/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/grounding_dino_sine_position_embedding.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-08:09:57:49
 * Description:
 *
 */

#ifndef __OPS_GROUNDING_DINO_SINE_POSITION_EMBEDDING_H__
#define __OPS_GROUNDING_DINO_SINE_POSITION_EMBEDDING_H__

#include <cmath>
#include <core/function.h>

namespace mariana {

struct GroundingDinoSinePositionEmbeddingFunc : public Function {
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
#if defined(MLM_USE_CUDA)
public:
    bool plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
#endif    
protected:
    const float m_scale         = 2*M_PI;
    float       m_temperature   = 1.f;
};

} // namespace mariana

#endif /* __OPS_GROUNDING_DINO_SINE_POSITION_EMBEDDING_H__ */

