/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : math.cu
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-19:05:15:49
 * Description:
 * 
 */

#include <ops/math.h>
#include <ops/backend/gpu/impl/math.h>

#include <core/node.h>
#include <core/backend/gpu/cuda_common.h>

namespace mariana {

bool AddFunc::plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (inputs.size() != 2) {
        MLOG(ERROR)<<"ADD inputs size must be 2";
        return false;
    }
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    cuda_set_device(cuda_ctx->device);
    if (outputs.empty()) {
        outputs.push_back(Tensor(DataOn::GPU));
    }
    outputs[0].try_realloc(inputs[0].dims(), inputs[0].dtype());
    return true;
}

bool AddFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    _parallel_sync(m_tp, inputs[0].total_size(), add_ele, std::ref(inputs[0]),
                   std::ref(inputs[1]), std::ref(outputs[0]), cuda_ctx);
    return true;
}

} // namespace mariana
