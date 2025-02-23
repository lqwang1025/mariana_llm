/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : models/qwen2.h
 * Authors    : lqwang@pandora
 * Create Time: 2025-02-15:10:09:55
 * Description:
 *
 */

#ifndef __QWEN2_H__
#define __QWEN2_H__

#include <vector>
#include <cstdint>
#include <core/tensor.h>
#include <models/lmodels.h>

namespace mariana {

class Qwen2 : public LModel {
public:
    Qwen2() {}
    virtual ~Qwen2() {}
    bool load_token(const char* dir_path);
    virtual AIResult compute(ExeContext& context)override;
    virtual bool make_graph(const char* dir_path, GptParams& gpt_params, ExeContext& context)override;
};

} // namespace mariana

#endif /* __QWEN2_H__ */

