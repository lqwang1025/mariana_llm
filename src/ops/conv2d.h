/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : conv2d.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-05-28:16:15:56
 * Description:
 *
 */

#ifndef __OPS_CONV2D_H__
#define __OPS_CONV2D_H__

#include <ops/ops.h>
#include <core/function.h>

namespace mariana {

struct ConvParam {
    uint8_t groups      = 1;
    uint8_t strides[2]  = {1, 1}; // x, y
    uint8_t padding[4]  = {0, 0, 0, 0}; // t,l,b,r
    uint8_t dilation[2] = {1, 1}; // x, y
    uint16_t kernel[4] = {0, 0, 0, 0}; // [oc, ic, kh, kw]
};

struct Conv2dFunc : public Function {
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
private:
    Tensor m_weight; // [oc, ic, kh, kw]
    Tensor m_bias;
    Tensor m_im2col;
    ConvParam m_param;
    bool m_output_trans;
    OpCategory m_act_cate = OpCategory::None;
};

} // namespace mariana

#endif /* __OPS_CONV2D_H__ */

