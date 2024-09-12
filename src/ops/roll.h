/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/roll.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-03:09:01:22
 * Description:
 *
 */

#ifndef __OPS_ROLL_H__
#define __OPS_ROLL_H__

#include <vector>
#include <core/function.h>

namespace mariana {

struct RollParam {
    std::vector<int32_t> dims;
    std::vector<int32_t> shifts;
};

struct RollFunc : public Function {
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
    RollParam param;
protected:
    friend class SwinLayerFunc;
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
    
};

} // namespace mariana

#endif /* __OPS_ROLL_H__ */

