/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/grounding_dino_encoder_before.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-10:09:13:05
 * Description:
 *
 */

#ifndef __OPS_GROUNDING_DINO_ENCODER_BEFORE_H__
#define __OPS_GROUNDING_DINO_ENCODER_BEFORE_H__

#include <core/function.h>

namespace mariana {

struct GroundingDinoEncoderBeforeFunc : public Function {
    struct SpatialShapes {
        uint32_t size = 0;
        uint32_t widths[8];
        uint32_t heights[8];
    };
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
private:
    Tensor m_level_embed;
    SpatialShapes m_spatial_shapes;
};

} // namespace mariana

#endif /* __OPS_GROUNDING_DINO_ENCODER_BEFORE_H__ */

