/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/math.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-21:18:23:21
 * Description:
 *
 */

#ifndef __OPS_ADD_H__
#define __OPS_ADD_H__

#include <core/function.h>

namespace mariana {

struct AddFunc : public Function {
    bool plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    friend class SwinLayerFunc;
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
};

struct MulFunc : public Function {
    bool plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
};

} // namespace mariana

#endif /* __OPS_ADD_H__ */

