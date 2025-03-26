/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : models/lmodels.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-18:10:48:22
 * Description:
 *
 */

#ifndef __MODELS_LMODELS_H__
#define __MODELS_LMODELS_H__

#include <cstdint>
#include <memory>
#include <functional>
#include <unordered_map>

#include <core/tensor.h>

#include <absl/types/any.h>

#include <models/model_param.h>
#include <utils/mariana_define.h>
#include <mariana_llm/mariana_llm.h>

namespace mariana {

struct ExeContext;
struct GptParams;
class Graph;
class Tokenizer;
class LmodelHolder;
struct ModelParam;
enum class LModelCategory : int16_t;
using AnyMap = std::unordered_map<std::string, ::absl::any>;
using SafeTensorsCallback = std::function<void(ModelParam::SafeTensorInfo&ti, ModelParam& param, const std::string&key)>;
using tensor_list = std::vector<Tensor>;

class LModel {
public:
    LModel() {}
    virtual ~LModel() {}
    virtual bool generate(ExeContext& context, AIResult& result);
    virtual bool make_graph(const char* dir_path, GptParams& gpt_params, ExeContext& context)=0;
    virtual bool init(const char* dir_path, GptParams& gpt_params, ExeContext& context) {
        TRACE();
        bool ok = make_graph(dir_path, gpt_params, context);
        ok = ok && _backend_setup(gpt_params, context);
        MLOG_IF(ERROR, !ok)<<"Lmodel init failed with:"<<dir_path;
        return ok;
    }
protected:
    virtual int32_t _sample(Tensor& logits, const std::vector<int32_t>& pre_ids, int offset = 0, int size = 0);
    virtual tensor_list _compute(ExeContext& context, const std::vector<int>& tokens, int32_t cache_len)=0;
    bool _load_safetensors(const char* safe_tensors, ModelParam& param,
                           SafeTensorsCallback callback =
                           [](ModelParam::SafeTensorInfo&sti, ModelParam& param, const std::string&key)->void {
                               param.sti_map[key] = sti;
                           });
    bool _backend_setup(GptParams& gpt_params, ExeContext& context);
protected:
    std::shared_ptr<Graph>     m_graph;
    std::shared_ptr<Tokenizer> m_tokenizer;
    float _repetition_penalty = 1.0f;
};

using LModelMake = std::function<LModel*()>;
DECLARE_SOMETHING_HOLDER(LModel, LModelCategory, LModelMake); // LModelHolder

} // namespace mariana

#endif /* __MODELS_LMODELS_H__ */

