/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : models/model_param.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-19:08:08:46
 * Description:
 *
 */

#ifndef __MODELS_MODEL_PARAM_H__
#define __MODELS_MODEL_PARAM_H__

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>

#include <ops/ops.h>
#include <core/data_type.h>

#include <mariana_llm/mariana_llm.h>

namespace mariana {

struct ModelParam {
    ModelParam(const ExeContext& ctx) {
        context = &ctx;
    }
    struct SafeTensorInfo {
        TypeMeta dtype;
        std::vector<int32_t> shape;
        std::vector<int32_t> data_offset;
        void* data{nullptr};
    };
    std::unordered_map<std::string, SafeTensorInfo> sti_map;
    int32_t           n_vocab                 = 0;
    int32_t           n_embd                  = 0;
    int32_t           n_head                  = 0;
    int32_t           n_layer                 = 0;
    int32_t           n_ff                    = 0;
    int32_t           patch_size              = 0;
    int32_t           pad_token_id            = 0;
    int32_t           max_position_embeddings = 0;
    int32_t           type_vocab_size         = 0;
    float             layer_norm_eps          = 0.f;
    // swin parameter
    int32_t           window_size             = 0;
    int32_t           shift_size              = 0;
    int32_t           patch_merge_step        = 0;
    // conv parameter
    int32_t           image_size              = 0;
    int32_t           i_channels              = 0;
    int32_t           o_channels              = 0;
    int32_t           kernel_size[2]          = {1, 1}; // x, y
    int32_t           strides[2]              = {1, 1}; // x, y
    int32_t           padding[4]              = {0, 0, 0, 0}; // t,l,b,r
    int32_t           dilation[2]             = {1, 1}; // x, y
    int32_t           groups                  = 1;
    bool              conv_output_trans       = false;
    OpCategory        act_cate                = OpCategory::None;
    // dino parameter
    int32_t positional_embedding_temperature  = 1;
    int32_t encoder_attention_heads           = 0;
    int32_t encoder_n_points                  = 0;
    int32_t num_feature_levels                = 0;
    int32_t d_model                           = 0;
    int32_t num_queries                       = 0;
    int32_t decoder_layers                    = 0;
    // permute parameter
    uint8_t           perms[4]                = {0, 0, 0, 0};
    const ExeContext* context                 = nullptr;
    void*             any_thing               = nullptr;
    bool              own_weight              = true; // If set true, the weigth that node own need be freed bt this node
};

} // namespace mariana

#endif /* __MODELS_MODEL_PARAM_H__ */

