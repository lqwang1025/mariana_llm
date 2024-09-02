/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/swin_patch_merging.cc
 * Authors    : lqwang@pandora
 * Create Time: 2024-07-05:05:40:03
 * Description:
 * 
 */

#include<cmath>

#include <utils/mariana_define.h>
#include <models/model_param.h>
#include <ops/swin_patch_merging.h>
#include <ops/backend/cpu/swin_patch_merging.h>

namespace mariana {

bool SwinPatchMergingFunc::init(const ModelParam& param, const std::string& node_name) {
    step = param.patch_merge_step;
    return true;
}

bool SwinPatchMergingFunc::plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (outputs.empty()) {
        outputs.push_back(Tensor(inputs[0].device()));
    }
    
    int32_t oh = ceil(float(context.runtime_info.feature_height)/float(step));
    int32_t ow = ceil(float(context.runtime_info.feature_width)/float(step));
    int32_t oc = inputs[0].dim_size() == 3 ? inputs[0].dim_at(2) : inputs[0].dim_at(3);
    outputs[0].try_realloc({inputs[0].dim_at(0), oh, ow, oc*step*step}, inputs[0].dtype());
    
    return true;
}

bool SwinPatchMergingFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    TRACE();
    Tensor input = inputs[0];
    input.reshape({input.dim_at(0), (int32_t)context.runtime_info.feature_height, (int32_t)context.runtime_info.feature_width, input.dim_at(2)});
    _parallel_sync(m_tp, outputs[0].total_size()/input.dim_at(3), swin_patch_merge, std::ref(input), std::ref(outputs[0]), step);
    return true;
}

} // namespace mariana
