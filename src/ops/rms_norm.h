/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/rms_norm.h
 * Authors    : lqwang@SMT23090002
 * Create Time: 2025-03-21:11:45:02
 * Description:
 *
 */

#ifndef __OPS_RMS_NORM_H__
#define __OPS_RMS_NORM_H__

#include <core/function.h>

namespace mariana {

struct RMSNormFunc : public Function {
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
    Tensor m_weight;
    float m_epsilon = 1e-6;
};

} // namespace mariana

#endif /* __OPS_RMS_NORM_H__ */
