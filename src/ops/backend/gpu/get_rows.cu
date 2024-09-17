/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : get_rows.cu
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-14:09:06:23
 * Description:
 * 
 */

#include <ops/get_rows.h>
#include <core/node.h>
#include <core/tensor_utils.h>
#include <core/backend/gpu/cuda_common.h>
#include <core/backend/gpu/cuda_allocator.h>
#include <ops/backend/gpu/impl/get_rows.h>

namespace mariana {

bool GetRowsFunc::plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (inputs[0].dim_size() != 2) {
        MLOG(ERROR)<<"GetRows input's dimision must be 2";
        return false;
    }
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    cuda_set_device(cuda_ctx->device);
    if (m_weight.device() != DataOn::GPU) {
        m_weight = m_weight.cuda(cuda_ctx->stream());
    }
    int32_t nb = inputs[0].dim_at(0);
    int32_t nr = inputs[0].dim_at(1);
    int32_t ne = m_weight.dim_at(1);
    if (outputs.empty()) {
        outputs.push_back(DataOn::GPU);
    }
    outputs[0].try_realloc({nb, nr, ne}, m_weight.dtype());
    return true;
}

bool GetRowsFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    _parallel_sync(m_tp, inputs[0].total_size(), get_rows, std::ref(inputs[0]),
                   std::ref(m_weight), std::ref(outputs[0]), cuda_ctx);
    return true;
}
    
} // namespace mariana
