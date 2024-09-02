/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/grounding_dino_for_detection.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-08-27:08:42:38
 * Description:
 *
 */

#ifndef __OPS_GROUNDING_DINO_FOR_DETECTION_H__
#define __OPS_GROUNDING_DINO_FOR_DETECTION_H__

#include <core/function.h>

namespace mariana {

struct MatMulFunc;

struct GroundingDinoForDetectionFunc : public Function {
    ~GroundingDinoForDetectionFunc();
    void set_thread_pool(ThreadPool* tp) override;
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
private:
    int32_t m_decoder_layers = 0;
    MatMulFunc* m_bbox_embed_func0 = nullptr;
    MatMulFunc* m_bbox_embed_func1 = nullptr;
    MatMulFunc* m_bbox_embed_func2 = nullptr;
    Tensor m_bbox_embed_out0;
    Tensor m_bbox_embed_out1;
};

} // namespace mariana

#endif /* __OPS_GROUNDING_DINO_FOR_DETECTION_H__ */

