/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/swin_layer.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-06-30:09:44:48
 * Description:
 *
 */

#ifndef __OPS_SWIN_LAYER_H__
#define __OPS_SWIN_LAYER_H__

#include <core/function.h>

namespace mariana {

struct MatMulFunc;
struct LayerNormFunc;
struct SliceFunc;
struct AddFunc;
struct RollFunc;
struct SelfAttentionFunc;

struct SwinParam {
    int32_t window_size = 0;
    int32_t shift_size  = 0;
};

struct SwinLayerFunc : public Function {
    ~SwinLayerFunc();
    virtual void set_thread_pool(ThreadPool* tp) override;
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
    bool _maybe_pad(uint32_t& pad_right, uint32_t& pad_bottom, uint32_t height, uint32_t width);
    void _create_attn_mask(const tensor_list& inputs, ExeContext& context);
private:
    SwinParam m_param;
    LayerNormFunc*     m_layer_norm_before = nullptr;
    SelfAttentionFunc* m_self_att          = nullptr;
    MatMulFunc*        m_self_att_omm      = nullptr;
    SliceFunc*         m_slice_func        = nullptr;
    AddFunc*           m_add_func          = nullptr;
    RollFunc*          m_roll_func         = nullptr;
    Tensor m_mask;
    Tensor m_attn_mask;
    Tensor m_lnb_out;
    Tensor m_pad_out;
    Tensor m_permute_out;
    Tensor m_self_att_out;
    Tensor m_self_att_omm_out;
    Tensor m_roll_out;
    uint32_t m_pad_right  = 0;
    uint32_t m_pad_bottom = 0;
};

} // namespace mariana

#endif /* __OPS_SWIN_LAYER_H__ */

