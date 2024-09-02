/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/att_mask.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-25:08:44:32
 * Description:
 * 
 */

#include <cfloat>

#include <ops/att_mask.h>
#include <utils/mariana_define.h>
#include <models/model_param.h>
#include <ops/sched_param.h>

namespace mariana {

bool AttMaskFunc::plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (outputs.empty()) {
        outputs.push_back(Tensor(inputs[0].device()));
    }
    outputs[0].try_realloc(inputs[0].dims(), TypeMeta::make<float>());
    return true;
}

static void __cast(SchedParam sched_param, const Tensor& input, const Tensor& out) {
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        uint8_t item = input.data_at<uint8_t>(i);
        *out.unsafe_ptr<float>(i) = item == 0 ? -FLT_MAX : 0;
    }
}

bool AttMaskFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    TRACE();
    _parallel_sync(m_tp, inputs[0].total_size(), __cast, std::ref(inputs[0]), std::ref(outputs[0]));
    return true;
}

} // namespace mariana

