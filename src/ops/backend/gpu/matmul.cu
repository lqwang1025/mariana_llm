/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/matmul.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-09-30:06:29:30
 * Description:
 * 
 */

#include <ops/matmul.h>
#include <core/node.h>
#include <core/backend/gpu/cuda_common.h>
#include <ops/backend/gpu/impl/matmul.h>

namespace mariana {

bool MatMulFunc::plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (inputs[0].dim_size() != 3) {
        MLOG(ERROR)<<"MATMUL input's dimision must be 3";
        return false;
    }
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    cuda_set_device(cuda_ctx->device);
    if (outputs.empty()) {
        outputs.push_back(Tensor(DataOn::GPU));
    }
    if (m_weight.device() != DataOn::GPU) {
        m_weight = m_weight.cuda(cuda_ctx->stream());
        m_bias = m_bias.cuda(cuda_ctx->stream());
    }
    int32_t nb = inputs[0].dim_at(0);
    int32_t nr = inputs[0].dim_at(1);
    int32_t nc = m_weight.dim_at(0);
    outputs[0].try_realloc({nb, nr, nc}, inputs[0].dtype());
    return true;
}

bool MatMulFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    _parallel_sync(m_tp, inputs[0].dim_at(0), matmul, std::ref(inputs[0]), std::ref(m_weight), std::ref(m_bias), std::ref(outputs[0]), 1.f, 1.f, m_act_cate, cuda_ctx);
    return true;
}

} // namespace mariana 
