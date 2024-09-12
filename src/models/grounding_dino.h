/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : models/grounding_dino.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-18:11:17:41
 * Description:
 *
 */

#ifndef __MODELS_GROUNDING_DINO_H__
#define __MODELS_GROUNDING_DINO_H__

#include <vector>
#include <cstdint>
#include <core/tensor.h>
#include <models/lmodels.h>

namespace mariana {

class GroundingDINO : public LModel {
public:
    GroundingDINO() {}
    virtual ~GroundingDINO() {}
    bool load_token(const char* dir_path);
    bool load_param(const char* dir_path, AnyMap& bert_param, AnyMap& dino_param,
                    AnyMap& swin_param, ModelParam& bert_model_param,
                    ModelParam& dino_model_param, ModelParam& swin_model_param);
    virtual AIResult compute(ExeContext& context)override;
    virtual bool make_graph(const char* dir_path, GptParams& gpt_params, ExeContext& context)override;
private:
    // reference in transformers/models/grounding_dino/modeling_grounding_dino.py:2020
    bool _generate_masks_with_special_tokens_and_transfer_map(const std::vector<int32_t>& input_ids,
                                                              std::vector<int32_t>& position_ids,
                                                              std::vector<uint8_t>& text_satt_mask);
    void _post_process(const Tensor& bbox, const Tensor& scores, const Tensor& probs, const std::vector<int32_t>& token, AIResult& result, ExeContext& context);
    void _pre_process(const Tensor& input, Tensor& out);
private:
    bool    m_do_normalize   = true;
    bool    m_do_pad         = true;
    bool    m_do_rescale     = true;
    bool    m_do_resize      = true;
    float   m_rescale_factor = 1.f;
    int32_t m_longest_edge   = 0;
    int32_t m_shortest_edge  = 0;
    int32_t m_patch_size     = 0;
    std::vector<float> m_means;
    std::vector<float> m_stds;
    
    Tensor m_image_tensor;
};

} // namespace mariana

#endif /* __MODELS_GROUNDING_DINO_H__ */

