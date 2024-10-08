/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/pass.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-07-11:06:00:49
 * Description:
 *
 */

#ifndef __OPS_PASS_H__
#define __OPS_PASS_H__

#include <core/function.h>

namespace mariana {

struct PassFunc : public Function {
    bool plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
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

#endif /* __OPS_PASS_H__ */

