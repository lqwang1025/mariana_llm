/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : function.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-18:07:10:40
 * Description:
 * 
 */

#include <core/node.h>
#include <core/function.h>

namespace mariana {

bool Function::on_plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
#if defined(MLM_USE_CUDA)
    //plan_forward_cpu(inputs, outputs, context);
    return plan_forward_gpu(inputs, outputs, context);
#else
    return plan_forward_cpu(inputs, outputs, context);
#endif
}

bool Function::on_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
#if defined(MLM_USE_CUDA)
    //return _forward_gpu(inputs, outputs, context);
    return _forward(inputs, outputs, context);
#else
    return _forward(inputs, outputs, context);
#endif
}

} // namespace mariana
