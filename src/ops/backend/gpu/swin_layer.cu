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
#include <ops/backend/gpu/impl/pad.h>
#include <ops/backend/gpu/impl/permute.h>

#include <core/node.h>
#include <core/impl/allocator.h>
#include <core/backend/gpu/cuda_common.h>

namespace mariana {

template<typename T>
__global__ void __rool_kernel(T* data, T value, int32_t nh_w, int32_t nw_w, int32_t data_stride_1, int32_t window_size, int32_t shift_size) {
    int32_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= nh_w*nw_w*window_size*window_size*window_size*window_size) return;
    int32_t idx = index;
    int32_t w2 = idx % window_size;
    idx /= window_size;
    int32_t w1 = idx % window_size;
    idx /= window_size;
    int32_t h2 = idx % window_size;
    idx /= window_size;
    int32_t h1 = idx % window_size;
    idx /= window_size;
    int32_t iw_w = idx % nw_w;
    idx /= nw_w;
    int32_t ih_w = idx;
    if ((iw_w == (nw_w-1)) && (ih_w < (nh_w-1))) {
        if ((h2 < window_size-shift_size && w2 > shift_size) ||
            (h2 > shift_size && w2 < window_size-shift_size)) {
            data[index] = value;
        }
    } else if ((ih_w == (nh_w-1)) && (iw_w < (nw_w-1))) {
        if ( (h1*window_size+h2 < (window_size-shift_size)*window_size &&
              w1*window_size+w2 > (window_size-shift_size)*window_size-1) ||
             (h1*window_size+h2 > (window_size-shift_size)*window_size-1 &&
              w1*window_size+w2 < (window_size-shift_size)*window_size) ) {
            data[index] = value;
        }
    } else if ((iw_w == (nw_w-1)) && (ih_w == (nh_w-1))) {
        if ( (h1*window_size+h2 < (window_size-shift_size)*window_size &&
              w1*window_size+w2 < (window_size-shift_size)*window_size) ||
             (h1*window_size+h2 > (window_size-shift_size)*window_size-1 &&
              w1*window_size+w2 > (window_size-shift_size)*window_size-1) ) {
            if ((h2 < window_size-shift_size && w2 > shift_size) ||
                (h2 > shift_size && w2 < window_size-shift_size)) {
                data[index] = value;
            }
        } else {
            data[index] = value;
        }
    } else {
        data[index] = 0;
    }
}

void SwinLayerFunc::_create_attn_mask_gpu(const tensor_list& inputs, ExeContext& context) {
    int32_t window_size = m_param.window_size;
    int32_t shift_size  = m_param.shift_size;
    uint32_t paded_h    = m_owner->info_shared_nodes()[0]->runtime_info().feature_height+m_pad_bottom;
    uint32_t paded_w    = m_owner->info_shared_nodes()[0]->runtime_info().feature_width+m_pad_right;
    uint32_t nh_w       = paded_h/window_size;
    uint32_t nw_w       = paded_w/window_size;
    if (m_attn_mask.total_size() == 0) {
        Tensor attn_mask(DataOn::GPU);
        m_attn_mask = attn_mask;
    }
    if (m_attn_mask.total_size() != 0 && (uint32_t)m_attn_mask.dim_at(0) == nh_w*nw_w &&
        (uint32_t)m_attn_mask.dim_at(2) == window_size*window_size) {
        return;
    }
    m_attn_mask.try_realloc({static_cast<int32_t>(nh_w*nw_w), 1,
            static_cast<int32_t>(window_size*window_size),
            static_cast<int32_t>(window_size*window_size)}, inputs[0].dtype());
    
#define CUDA_ATTNMASK_BLOCK_SIZE 256
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    float* data = m_attn_mask.unsafe_ptr<float>(0);
    __rool_kernel<float><<<get_cuda_gridsize(m_attn_mask.total_size(), CUDA_ATTNMASK_BLOCK_SIZE),
        CUDA_ATTNMASK_BLOCK_SIZE, 0, cuda_ctx->stream(0)>>>(data, -100.f, nh_w, nw_w, m_attn_mask.stride_at(1), window_size, shift_size);
    cuda_ctx->stream_sync(cuda_ctx->stream(0));
}

bool SwinLayerFunc::plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    cuda_set_device(cuda_ctx->device);
    if (m_mask.device() != DataOn::GPU) {
        m_mask = m_mask.cuda(cuda_ctx->stream());;
    }
    if (m_lnb_out.total_size() == 0) {
        m_lnb_out = Tensor(DataOn::GPU);
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
    
    if (m_self_att_out.total_size() == 0) {
        m_self_att_out = Tensor(DataOn::GPU);
    }
    __outputs = {m_self_att_out};
    tensor_list __inputs = {m_permute_out};
    m_self_att->plan_forward_gpu(__inputs, __outputs, context);
    m_permute_out.reshape({dim1, dim2, dim3, dim4, dim5, dim6});
    
    __inputs  = {m_self_att_out};
    if (m_self_att_omm_out.total_size() == 0) {
        m_self_att_omm_out = Tensor(DataOn::GPU);
    }
    __outputs = {m_self_att_omm_out};
    m_self_att_omm->plan_forward_gpu(__inputs, __outputs, context);
    if (m_param.shift_size > 0) {
        _create_attn_mask_gpu(inputs, context);
    }
    return true;
}

bool SwinLayerFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    // 1. laynorm before
    tensor_list __outputs = {m_lnb_out};
    m_layer_norm_before->_forward_gpu(inputs, __outputs, context);

    m_lnb_out.reshape({inputs[0].dim_at(0),
            static_cast<int32_t>(m_owner->info_shared_nodes()[0]->runtime_info().feature_height),
            static_cast<int32_t>(m_owner->info_shared_nodes()[0]->runtime_info().feature_width),
            inputs[0].dim_at(2)});
    // 1.1 pad maybe
    Tensor __route;
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    if (m_pad_bottom != 0 || m_pad_right != 0) {
        uint32_t padding[6] = {0, 0, 0, m_pad_right, 0, m_pad_bottom};
        _parallel_sync(m_tp, m_pad_out.dim_at(0), nchw_pad, std::ref(m_lnb_out), std::ref(m_pad_out), padding, 0.f, cuda_ctx);
        __route = m_pad_out;
    } else {
        __route = m_lnb_out;
    }

    tensor_list __inputs;
    // 1.2 roll maybe
    if (m_param.shift_size > 0) {
        __inputs = {__route};
        if (m_roll_out.total_size() == 0) {
            m_roll_out = Tensor(DataOn::GPU);
        }
        __outputs = {m_roll_out};
        m_roll_func->param.dims   = {1, 2};
        m_roll_func->param.shifts = {-m_param.shift_size, -m_param.shift_size};
        m_roll_func->plan_forward_gpu(__inputs, __outputs, context);
        m_roll_func->_forward_gpu(__inputs, __outputs, context);
        __route = m_roll_out;
    }

    // 2. partition windows
    __route.reshape({__route.dim_at(0), __route.dim_at(1)/m_param.window_size, m_param.window_size,
            __route.dim_at(2)/m_param.window_size, m_param.window_size, __route.dim_at(3)});
    uint8_t perms[6] = {0, 1, 3, 2, 4, 5};
    _parallel_sync(m_tp, 1, permute6, std::ref(__route), std::ref(m_permute_out), perms, cuda_ctx);

    
    // 3. attention
    m_permute_out.reshape({m_permute_out.dim_at(1)*m_permute_out.dim_at(2), m_permute_out.dim_at(3)*m_permute_out.dim_at(4), m_permute_out.dim_at(5)});
    __inputs = {m_permute_out, m_mask};
    if (m_param.shift_size > 0) {
        __inputs.push_back(m_attn_mask);
    }
    __outputs = {m_self_att_out};
    m_self_att->_forward_gpu(__inputs, __outputs, context);
    
    // 3.1 attention projection
    __inputs = {m_self_att_out};
    __outputs = {m_self_att_omm_out};
    m_self_att_omm->_forward_gpu(__inputs, __outputs, context);

    
    // 4. windows split
    int32_t padh = m_pad_bottom+m_owner->info_shared_nodes()[0]->runtime_info().feature_height;
    int32_t padw = m_pad_right+m_owner->info_shared_nodes()[0]->runtime_info().feature_width;
    m_self_att_omm_out.reshape({1, padh/m_param.window_size, padw/m_param.window_size, m_param.window_size, m_param.window_size, m_self_att_omm_out.dim_at(2)});
    
    m_permute_out.reshape({m_self_att_omm_out.dim_at(0), m_self_att_omm_out.dim_at(1),
            m_self_att_omm_out.dim_at(3), m_self_att_omm_out.dim_at(2), m_self_att_omm_out.dim_at(4), m_self_att_omm_out.dim_at(5)});
    _parallel_sync(m_tp, m_self_att_omm_out.total_size(), permute6, std::ref(m_self_att_omm_out), std::ref(m_permute_out), perms, cuda_ctx);

    m_permute_out.reshape({m_permute_out.dim_at(0), m_permute_out.dim_at(1)*m_permute_out.dim_at(2),
            m_permute_out.dim_at(3)*m_permute_out.dim_at(4), m_permute_out.dim_at(5)});
    
    // todo
    if (m_param.shift_size > 0) {
        __inputs = {m_permute_out};
        __outputs = {m_roll_out};
        m_roll_func->param.dims   = {1, 2};
        m_roll_func->param.shifts = {m_param.shift_size, m_param.shift_size};
        m_roll_func->plan_forward_gpu(__inputs, __outputs, context);
        m_roll_func->_forward_gpu(__inputs, __outputs, context);
        __route = m_roll_out;
    } else {
        __route = m_permute_out;
    }

    if (m_pad_bottom != 0 || m_pad_right != 0) {
        // 4.1 slice to ori feature map
        m_slice_func->param.starts = {0, 0};
        m_slice_func->param.ends = {(int32_t)m_owner->info_shared_nodes()[0]->runtime_info().feature_height,
            (int32_t)m_owner->info_shared_nodes()[0]->runtime_info().feature_width};
        m_slice_func->param.axes = {1, 2};
        m_slice_func->param.steps = {1, 1};
        __inputs = {__route};
        __outputs = {m_lnb_out};
        m_slice_func->plan_forward_gpu(__inputs, __outputs, context);
        m_slice_func->_forward_gpu(__inputs, __outputs, context);
        m_lnb_out.reshape({m_lnb_out.dim_at(0), m_lnb_out.dim_at(1)*m_lnb_out.dim_at(2), m_lnb_out.dim_at(3)});

        // 4.2 shortcut
        __inputs = {m_lnb_out, inputs[0]};
        m_add_func->plan_forward_gpu(__inputs, outputs, context);
        m_add_func->_forward_gpu(__inputs, outputs, context);
    } else {
        // 4.2 shortcut
        __route.reshape({__route.dim_at(0), __route.dim_at(1)*__route.dim_at(2), __route.dim_at(3)});
        __inputs = {__route, inputs[0]};
        m_add_func->plan_forward_gpu(__inputs, outputs, context);
        m_add_func->_forward_gpu(__inputs, outputs, context);
    }
    
    return true;
}

} // namespace mariana
