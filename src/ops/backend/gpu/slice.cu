/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/slice.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-10:05:39:02
 * Description:
 * 
 */

#include <ops/slice.h>
#include <ops/backend/gpu/impl/slice.h>

#include <core/node.h>
#include <core/backend/gpu/cuda_common.h>

namespace mariana {

bool SliceFunc::plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    cuda_set_device(cuda_ctx->device);
    if (outputs.empty()) {
        outputs.push_back(Tensor(DataOn::GPU));
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

bool SliceFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    _parallel_sync(m_tp, 1, slice4, std::ref(inputs[0]), std::ref(outputs[0]), param, cuda_ctx);
    return true;
}

} // namespace mariana
