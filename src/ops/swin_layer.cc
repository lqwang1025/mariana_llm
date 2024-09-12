/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : swin_layer.cc
 * Authors    : lqwang@pandora
 * Create Time: 2024-06-30:09:45:52
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
#include <ops/backend/cpu/pad.h>
#include <ops/backend/cpu/permute.h>
#include <ops/backend/cpu/get_rows.h>
#include <core/impl/thread_pool.h>
#include <core/impl/allocator.h>

#include <models/model_param.h>
#include <core/node.h>
#include <core/function.h>
#include <core/tensor_utils.h>

namespace mariana {

SwinLayerFunc::~SwinLayerFunc() {
    delete m_layer_norm_before;
    delete m_self_att;
    delete m_self_att_omm;
    delete m_slice_func;
    delete m_add_func;
    delete m_roll_func;
}

void SwinLayerFunc::set_thread_pool(ThreadPool* tp) {
    m_tp = tp;
    m_layer_norm_before->set_thread_pool(tp);
    m_self_att->set_thread_pool(tp);
    m_self_att_omm->set_thread_pool(tp);
    m_slice_func->set_thread_pool(tp);
    m_add_func->set_thread_pool(tp);
    m_roll_func->set_thread_pool(tp);
}

bool SwinLayerFunc::init(const ModelParam& param, const std::string& node_name) {
    m_param.window_size = param.window_size;
    m_param.shift_size  = param.shift_size;
    m_layer_norm_before = new LayerNormFunc{};
    m_self_att          = new SelfAttentionFunc{};
    m_self_att_omm      = new MatMulFunc{};
    m_slice_func        = new SliceFunc{};
    m_add_func          = new AddFunc{};
    m_roll_func         = new RollFunc{};
    
    m_layer_norm_before->init(param, node_name+".layernorm_before");
    m_self_att->init(param, node_name+".attention.self");
    m_self_att_omm->init(param, node_name+".attention.output.dense");
    
    ModelParam::SafeTensorInfo sti;
    TRY_STL(sti = param.sti_map.at(node_name+".attention.self.relative_position_bias_table"), return false); // relative_position_index
    Tensor mask_table(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    TRY_STL(sti = param.sti_map.at(node_name+".attention.self.relative_position_index"), return false);
    Tensor mask_index(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    mask_index.reshape({1, mask_index.dim_at(0)*mask_index.dim_at(1)});
    Tensor mask_out(mask_index.device());
    mask_out.try_realloc({1, mask_index.dim_at(1), mask_table.dim_at(1)}, mask_table.dtype());
    
    ThreadPool* tp = new ThreadPool(ThreadPool::default_num_threads());
    _parallel_sync(tp, mask_index.total_size(), get_rows, std::ref(mask_index),
                   std::ref(mask_table), std::ref(mask_out));
    mask_out.reshape({1, m_param.window_size*m_param.window_size, m_param.window_size*m_param.window_size, mask_table.dim_at(1)});
    
    Tensor mask(mask_index.device());
    mask.try_realloc({1, mask_table.dim_at(1), m_param.window_size*m_param.window_size, m_param.window_size*m_param.window_size}, mask_out.dtype());
    m_mask = mask;
    uint8_t perms[4] = {0, 3, 1, 2};
    _parallel_sync(tp, mask_out.total_size(), permute4, std::ref(mask_out), std::ref(m_mask), perms);
    delete tp;
    return true;
}

void SwinLayerFunc::_create_attn_mask(const tensor_list& inputs, ExeContext& context) {
    int32_t window_size = m_param.window_size;
    int32_t shift_size  = m_param.shift_size;
    uint32_t paded_h    = m_owner->info_shared_nodes()[0]->runtime_info().feature_height+m_pad_bottom;
    uint32_t paded_w    = m_owner->info_shared_nodes()[0]->runtime_info().feature_width+m_pad_right;
    uint32_t nh_w       = paded_h/window_size;
    uint32_t nw_w       = paded_w/window_size;
    Tensor attn_mask(inputs[0].device());
    m_attn_mask = attn_mask;
    if (m_attn_mask.total_size() != 0 && (uint32_t)m_attn_mask.dim_at(0) == nh_w*nw_w &&
        (uint32_t)m_attn_mask.dim_at(2) == window_size*window_size) {
        return;
    }
    m_attn_mask.try_realloc({static_cast<int32_t>(nh_w*nw_w), 1,
            static_cast<int32_t>(window_size*window_size),
            static_cast<int32_t>(window_size*window_size)}, inputs[0].dtype());
    IAllocator* allocator = get_allocator(m_attn_mask.device());
    allocator->memset(m_attn_mask.unsafe_ptr<float>(0), 0, m_attn_mask.total_size()*m_attn_mask.dtype().itemsize());
    auto roolh_func = [&](float* data, float value)->void {
        for (int32_t h1 = 0; h1 < window_size; ++h1) {
            for (int32_t h2 = 0; h2 < window_size; ++h2) {
                size_t h_offset = h1*window_size+h2;
                h_offset = h_offset*window_size*window_size;
                for (int32_t w1 = 0; w1 < window_size; ++w1) {
                    for (int32_t w2 = 0; w2 < window_size; ++w2) {
                        if ((h2 < window_size-shift_size && w2 > shift_size) ||
                            (h2 > shift_size && w2 < window_size-shift_size)) {
                            data[h_offset+w1*window_size+w2] = value;
                        }
                    }
                }
            }
        }
    };

    auto roolw_func = [&](float* data, float value)->void {
        for (int32_t h1 = 0; h1 < window_size; ++h1) {
            for (int32_t h2 = 0; h2 < window_size; ++h2) {
                size_t h_offset = h1*window_size+h2;
                h_offset = h_offset*window_size*window_size;
                for (int32_t w1 = 0; w1 < window_size; ++w1) {
                    for (int32_t w2 = 0; w2 < window_size; ++w2) {
                        if ( (h1*window_size+h2 < (window_size-shift_size)*window_size &&
                              w1*window_size+w2 > (window_size-shift_size)*window_size-1) ||
                             (h1*window_size+h2 > (window_size-shift_size)*window_size-1 &&
                              w1*window_size+w2 < (window_size-shift_size)*window_size) ) {
                            data[h_offset+w1*window_size+w2] = value;
                        }
                    }
                }
            }
        }
    };

    auto roolhw_func = [&](float* data, float value)->void {
        for (int32_t h1 = 0; h1 < window_size; ++h1) {
            for (int32_t h2 = 0; h2 < window_size; ++h2) {
                size_t h_offset = h1*window_size+h2;
                h_offset = h_offset*window_size*window_size;
                for (int32_t w1 = 0; w1 < window_size; ++w1) {
                    for (int32_t w2 = 0; w2 < window_size; ++w2) {
                        if ( (h1*window_size+h2 < (window_size-shift_size)*window_size &&
                              w1*window_size+w2 < (window_size-shift_size)*window_size) ||
                             (h1*window_size+h2 > (window_size-shift_size)*window_size-1 &&
                              w1*window_size+w2 > (window_size-shift_size)*window_size-1) ) {
                            if ((h2 < window_size-shift_size && w2 > shift_size) ||
                                (h2 > shift_size && w2 < window_size-shift_size)) {
                                data[h_offset+w1*window_size+w2] = value;
                            }
                        } else {
                            data[h_offset+w1*window_size+w2] = value;
                        }
                    }
                }
            }
        }
    };

    for (uint32_t h = 0; h < nh_w-1; ++h) {
        uint32_t offset = h*nw_w+(nw_w-1);
        float* data = m_attn_mask.unsafe_ptr<float>(offset*m_attn_mask.stride_at(1));
        roolh_func(data, -100.f);
    }
    
    for (uint32_t w = 0; w < nw_w-1; ++w) {
        uint32_t offset = (nh_w-1)*nw_w+w;
        float* data = m_attn_mask.unsafe_ptr<float>(offset*m_attn_mask.stride_at(1));
        roolw_func(data, -100.f);
    }
    uint32_t offset = (nh_w-1)*nw_w+(nw_w-1);
    float* data = m_attn_mask.unsafe_ptr<float>(offset*m_attn_mask.stride_at(1));
    roolhw_func(data, -100.f);
}

bool SwinLayerFunc::_maybe_pad(uint32_t& pad_right, uint32_t& pad_bottom, uint32_t height, uint32_t width) {
    pad_right = (m_param.window_size - width%m_param.window_size)%m_param.window_size;
    pad_bottom = (m_param.window_size - height%m_param.window_size)%m_param.window_size;
    if (pad_right > 0 || pad_bottom > 0) {
        return true;
    } else {
        return false;
    }
}

bool SwinLayerFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    tensor_list __outputs = {m_lnb_out};
    m_layer_norm_before->plan_forward_cpu(inputs, __outputs, context);

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
    m_self_att->plan_forward_cpu(__inputs, __outputs, context);
    m_permute_out.reshape({dim1, dim2, dim3, dim4, dim5, dim6});
    
    __inputs  = {m_self_att_out};
    __outputs = {m_self_att_omm_out};
    m_self_att_omm->plan_forward_cpu(__inputs, __outputs, context);
    if (m_param.shift_size > 0) {
        _create_attn_mask(inputs, context);
    }
    return true;
}

bool SwinLayerFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {// maybe optimized by cancel the padding 
    // 1. laynorm before
    tensor_list __outputs = {m_lnb_out};
    
    m_layer_norm_before->_forward(inputs, __outputs, context);
    m_lnb_out.reshape({inputs[0].dim_at(0),
            static_cast<int32_t>(m_owner->info_shared_nodes()[0]->runtime_info().feature_height),
            static_cast<int32_t>(m_owner->info_shared_nodes()[0]->runtime_info().feature_width),
            inputs[0].dim_at(2)});
    // 1.1 pad maybe
    Tensor __route;
    if (m_pad_bottom != 0 || m_pad_right != 0) {
        uint32_t padding[6] = {0, 0, 0, m_pad_right, 0, m_pad_bottom};
        _parallel_sync(m_tp, m_pad_out.total_size(), nchw_pad, std::ref(m_lnb_out), std::ref(m_pad_out), padding, 0.f);
        __route = m_pad_out;
    } else {
        __route = m_lnb_out;
    }

    tensor_list __inputs;
    // 1.2 roll maybe
    if (m_param.shift_size > 0) {
        __inputs = {__route};
        __outputs = {m_roll_out};
        m_roll_func->param.dims   = {1, 2};
        m_roll_func->param.shifts = {-m_param.shift_size, -m_param.shift_size};
        m_roll_func->plan_forward_cpu(__inputs, __outputs, context);
        m_roll_func->_forward(__inputs, __outputs, context);
        __route = m_roll_out;
    }
    // 2. partition windows
    __route.reshape({__route.dim_at(0), __route.dim_at(1)/m_param.window_size, m_param.window_size,
            __route.dim_at(2)/m_param.window_size, m_param.window_size, __route.dim_at(3)});
    uint8_t perms[6] = {0, 1, 3, 2, 4, 5};
    _parallel_sync(m_tp, __route.total_size(), permute6, std::ref(__route), std::ref(m_permute_out), perms);
    // 3. attention
    m_permute_out.reshape({m_permute_out.dim_at(1)*m_permute_out.dim_at(2), m_permute_out.dim_at(3)*m_permute_out.dim_at(4), m_permute_out.dim_at(5)});
    __inputs = {m_permute_out, m_mask};
    if (m_param.shift_size > 0) {
        __inputs.push_back(m_attn_mask);
    }
    __outputs = {m_self_att_out};
    m_self_att->_forward(__inputs, __outputs, context);
    // 3.1 attention projection
    __inputs = {m_self_att_out};
    __outputs = {m_self_att_omm_out};
    m_self_att_omm->_forward(__inputs, __outputs, context);
    // 4. windows split
    int32_t padh = m_pad_bottom+m_owner->info_shared_nodes()[0]->runtime_info().feature_height;
    int32_t padw = m_pad_right+m_owner->info_shared_nodes()[0]->runtime_info().feature_width;
    m_self_att_omm_out.reshape({1, padh/m_param.window_size, padw/m_param.window_size, m_param.window_size, m_param.window_size, m_self_att_omm_out.dim_at(2)});
    
    m_permute_out.reshape({m_self_att_omm_out.dim_at(0), m_self_att_omm_out.dim_at(1),
            m_self_att_omm_out.dim_at(3), m_self_att_omm_out.dim_at(2), m_self_att_omm_out.dim_at(4), m_self_att_omm_out.dim_at(5)});
    _parallel_sync(m_tp, m_self_att_omm_out.total_size(), permute6, std::ref(m_self_att_omm_out), std::ref(m_permute_out), perms);
    m_permute_out.reshape({m_permute_out.dim_at(0),
            m_permute_out.dim_at(1)*m_permute_out.dim_at(2),
            m_permute_out.dim_at(3)*m_permute_out.dim_at(4),
            m_permute_out.dim_at(5)});
    // todo
    if (m_param.shift_size > 0) {
        __inputs = {m_permute_out};
        __outputs = {m_roll_out};
        m_roll_func->param.dims   = {1, 2};
        m_roll_func->param.shifts = {m_param.shift_size, m_param.shift_size};
        m_roll_func->plan_forward_cpu(__inputs, __outputs, context);
        m_roll_func->_forward(__inputs, __outputs, context);
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
        m_slice_func->plan_forward_cpu(__inputs, __outputs, context);
        m_slice_func->_forward(__inputs, __outputs, context);
        m_lnb_out.reshape({m_lnb_out.dim_at(0), m_lnb_out.dim_at(1)*m_lnb_out.dim_at(2), m_lnb_out.dim_at(3)});
        // 4.2 shortcut
        __inputs = {m_lnb_out, inputs[0]};
        m_add_func->plan_forward_cpu(__inputs, outputs, context);
        m_add_func->_forward(__inputs, outputs, context);
    } else {
        // 4.2 shortcut
        __route.reshape({__route.dim_at(0), __route.dim_at(1)*__route.dim_at(2), __route.dim_at(3)});
        __inputs = {__route, inputs[0]};
        m_add_func->plan_forward_cpu(__inputs, outputs, context);
        m_add_func->_forward(__inputs, outputs, context);
    }
    return true;
}

} // namespace mariana
