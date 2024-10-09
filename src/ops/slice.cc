/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/slice.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-02:11:17:39
 * Description:
 * 
 */

#include <ops/slice.h>
#include <ops/backend/cpu/slice.h>
#include <models/model_param.h>
#include <utils/mariana_define.h>

namespace mariana {

bool SliceFunc::init(const ModelParam& param, const std::string& node_name) {
    MLOG(FATAL)<<"TODO: SliceFunc is an internal operator!!!";
    return true;
}

bool SliceFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (outputs.empty()) {
        outputs.push_back(Tensor(inputs[0].device()));
    }
    std::vector<int32_t> oshape;
    oshape.resize(inputs[0].dim_size());
    for (size_t ii = 0; ii < inputs[0].dim_size(); ++ii) {
        int32_t dim = inputs[0].dim_at(ii);
        for (size_t i = 0; i < param.axes.size(); ++i) {
            if (param.axes[i] == (int32_t)ii) {
                dim = param.ends[i] - param.starts[i];
                dim = (dim - 1)/param.steps[i] + 1;
            }
        }
        oshape[ii] = dim;
    }
    outputs[0].try_realloc(oshape, inputs[0].dtype());
    return true;
}

bool SliceFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    TRACE();
    _parallel_sync(m_tp, outputs[0].total_size(), slice4, std::ref(inputs[0]), std::ref(outputs[0]), param);
    return true;
}

} // namespace mariana


