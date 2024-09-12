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

class LModel {
public:
    LModel() {}
    virtual ~LModel() {}
    virtual AIResult compute(ExeContext& context)=0;
    virtual bool make_graph(const char* dir_path, GptParams& gpt_params, ExeContext& context)=0;
    virtual bool init(const char* dir_path, GptParams& gpt_params, ExeContext& context) {
        TRACE();
        bool ok = make_graph(dir_path, gpt_params, context);
        ok = ok && _backend_setup(gpt_params, context);
        MLOG_IF(ERROR, !ok)<<"Lmodel init failed with:"<<dir_path;
        return ok;
    }
protected:
    bool _load_safetensors(const char* safe_tensors, ModelParam& param,
                           SafeTensorsCallback callback =
                           [](ModelParam::SafeTensorInfo&sti, ModelParam& param, const std::string&key)->void {
                               param.sti_map[key] = sti;
                           });
    bool _backend_setup(GptParams& gpt_params, ExeContext& context);
    bool _load_config(const char* config_file, AnyMap& any_map);
protected:
    std::shared_ptr<Graph>     m_graph;
    std::shared_ptr<Tokenizer> m_tokenizer;
};

using LModelMake = std::function<LModel*()>;
DECLARE_SOMETHING_HOLDER(LModel, LModelCategory, LModelMake); // LModelHolder

} // namespace mariana

#endif /* __MODELS_LMODELS_H__ */

