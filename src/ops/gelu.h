/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/gelu.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-25:16:49:44
 * Description:
 *
 */

#ifndef __OPS_GELU_H__
#define __OPS_GELU_H__

#include <core/function.h>

namespace mariana {

struct GELUFunc : public Function {
    bool plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    friend class SwinLayerFunc;
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
};

} // namespace mariana

#endif /* __OPS_GELU_H__ */

