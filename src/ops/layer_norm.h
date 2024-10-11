/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/layer_norm.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-18:13:21:27
 * Description:
 *
 */

#ifndef __OPS_LAYER_NORM_H__
#define __OPS_LAYER_NORM_H__

#include <core/function.h>

namespace mariana {

struct NormParam {
    float epsilon;
    int8_t axies;
    uint8_t groups;
};

struct LayerNormFunc : public Function {
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    friend class SwinLayerFunc;
    friend class SwinStageOutputFunc;
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
#if defined(MLM_USE_CUDA)
public:
    bool plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
#endif

private:
    Tensor m_weight;
    Tensor m_bias;
    float m_epsilon = 1e-5;
    int8_t m_axies = -1;
};

} // namespace mariana

#endif /* __OPS_LAYER_NORM_H__ */

