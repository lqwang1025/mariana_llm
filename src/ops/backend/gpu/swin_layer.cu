/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/swin_layer.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-08:19:39:02
 * Description:
 * 
 */

#include <ops/swin_layer.h>
#include <ops/layer_norm.h>
#include <ops/matmul.h>
#include <ops/slice.h>
#include <ops/math.h>
#include <ops/roll.h>
#include <ops/self_attention.h>

#include <core/node.h>
#include <core/backend/gpu/cuda_common.h>

namespace mariana {

bool SwinLayerFunc::plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    cuda_set_device(cuda_ctx->device);
    if (m_mask.device() != DataOn::GPU) {
        m_mask = m_mask.cuda(cuda_ctx->stream());;
    }
    tensor_list __outputs = {m_lnb_out};
    m_layer_norm_before->plan_forward_gpu(inputs, __outputs, context);
    int32_t dim1 = m_lnb_out.dim_at(0);
    int32_t dim2 = m_owner->info_shared_nodes()[0]->runtime_info().feature_height/m_param.window_size;
    int32_t dim3 = m_owner->info_shared_nodes()[0]->runtime_info().feature_width/m_param.window_size;
    int32_t dim4 = m_param.window_size;
    int32_t dim5 = m_param.window_size;
    int32_t dim6 = m_lnb_out.dim_at(2);
    bool yes = _maybe_pad(m_pad_right, m_pad_bottom, m_owner->info_shared_nodes()[0]->runtime_info().feature_height, m_owner->info_shared_nodes()[0]->runtime_info().feature_width);
    if (yes) {
        if (m_pad_out.total_size() == 0) {
            m_pad_out     = Tensor(m_lnb_out.device());
        }
        m_pad_out.try_realloc({m_lnb_out.dim_at(0),
                static_cast<int32_t>(m_owner->info_shared_nodes()[0]->runtime_info().feature_height+m_pad_bottom),
                static_cast<int32_t>(m_owner->info_shared_nodes()[0]->runtime_info().feature_width+m_pad_right),
                m_lnb_out.dim_at(2)}, m_lnb_out.dtype());
        dim2 = m_pad_out.dim_at(1)/m_param.window_size;
        dim3 = m_pad_out.dim_at(2)/m_param.window_size;
        dim4 = m_param.window_size;
        dim5 = m_param.window_size;
        dim6 = m_pad_out.dim_at(3);
    }
    if (m_permute_out.total_size() == 0) {
        m_permute_out = Tensor(m_pad_out.device());
    }
    m_permute_out.try_realloc({dim2*dim3, dim4*dim5, dim6}, m_pad_out.dtype());
    __outputs = {m_self_att_out};
    tensor_list __inputs = {m_permute_out};
    m_self_att->plan_forward_gpu(__inputs, __outputs, context);
    m_permute_out.reshape({dim1, dim2, dim3, dim4, dim5, dim6});
    return true;
}

bool SwinLayerFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    return true;
}

} // namespace mariana
