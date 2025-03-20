/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/rope.h
 * Authors    : lqwang@SMT23090002
 * Create Time: 2025-03-17:09:07:47
 * Description:
 *
 */

#ifndef __OPS_ROPE_H__
#define __OPS_ROPE_H__

#include <vector>
#include <core/function.h>

namespace mariana {

struct ROPEParam {
    float              rope_theta            = 1.f;
    float              partial_rotary_factor = 1.f;
    std::string        rope_type             = "";
    float              attention_factor      = 1.f;
    Tensor inv_freq;
};

struct ROPEFunc : public Function {
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
    ROPEParam param;
private:
    Tensor _compute_default_rope_parameters(const ModelParam& param);
protected:
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
#if defined(MLM_USE_CUDA)
public:
    bool plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
#endif
};

} // namespace mariana

#endif /* __OPS_ROPE_H__ */

