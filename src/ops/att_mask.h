/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/att_mask.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-25:08:42:57
 * Description:
 *
 */

#ifndef __OPS_MASK_H__
#define __OPS_MASK_H__

#include <core/function.h>

namespace mariana {

struct AttMaskFunc : public Function {
    bool plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
};

} // namespace mariana

#endif /* __OPS_MASK_H__ */

