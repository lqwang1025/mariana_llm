/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/swin_patch_merging.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-07-05:05:38:55
 * Description:
 *
 */

#ifndef __OPS_SWIN_PATCH_MERGING_H__
#define __OPS_SWIN_PATCH_MERGING_H__

#include <core/function.h>

namespace mariana {

struct SwinPatchMergingFunc : public Function {
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
    int32_t step = 1;
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

#endif /* __OPS_SWIN_PATCH_MERGING_H__ */

