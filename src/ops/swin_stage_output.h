/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/swin_stage_output.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-05:15:42:11
 * Description:
 *
 */

#ifndef __OPS_SWIN_STAGE_OUTPUT_H__
#define __OPS_SWIN_STAGE_OUTPUT_H__

#include <vector>
#include <core/function.h>

namespace mariana {

struct LayerNormFunc;

struct SwinStageOutputFunc : public Function {
    ~SwinStageOutputFunc();
    bool init(const ModelParam& param, const std::string& node_name)override;
    void set_thread_pool(ThreadPool* tp)override;
    bool plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
    LayerNormFunc* m_layer_norm = nullptr;
    Tensor m_ln_out;
#if defined(MLM_USE_CUDA)
public:
    bool plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
#endif
};

} // namespace mariana

#endif /* __OPS_SWIN_STAGE_OUTPUT_H__ */

