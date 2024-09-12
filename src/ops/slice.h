/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/slice.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-02:11:12:06
 * Description:
 *
 */

#ifndef __OPS_SLICE_H__
#define __OPS_SLICE_H__

#include <vector>
#include <core/function.h>

namespace mariana {

struct SliceParam {
    std::vector<int32_t> starts;
    std::vector<int32_t> ends;
    std::vector<int32_t> axes;
    std::vector<int32_t> steps;
};

struct SliceFunc : public Function {
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
    SliceParam param;
protected:
    friend class SwinLayerFunc;
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
    
};

} // namespace mariana

#endif /* __OPS_SLICE_H__ */

