/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : models/lmodels.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-18:12:52:27
 * Description:
 * 
 */

#include <fstream>
#include <functional>
#include <unordered_set>

#include <utils/sys.h>

#include <models/lmodels.h>
#include <core/tensor_utils.h>

#include <core/graph.h>
#include <core/impl/allocator.h>
#include <core/device_type.h>

#include <utils/mariana_define.h>
#include <utils/dtype_utils.h>
#include <utils/rapidjson/document.h>

#include <token/sentencepiece.h>

namespace mariana {

bool LModel::_load_safetensors(const char* safe_tensors, ModelParam& param, SafeTensorsCallback callback) {
    TRACE();
    if (false == file_exist(safe_tensors)) {
        MLOG(ERROR)<<"model.safetensors is not exist in:"<<safe_tensors;
        return false;
    }
    
    std::ifstream file(safe_tensors, std::ios::binary | std::ios::in);
    
    if (!file.is_open()) {
        MLOG(ERROR)<<"Open "<<safe_tensors<<" failed";
        return false;
    }
    uint64_t header_size;
    file.read(reinterpret_cast<char*>(&header_size), 8);
    std::string header;
    header.resize(header_size);
    file.read(header.data(), header_size);
    rapidjson::Document doc;
	doc.Parse(header.c_str());
    if (doc.HasParseError()) {
        MLOG(ERROR)<<"Parse json file failed";
        return false;
    }
    //   "layer_name": {
    //   "dtype": "I64",
    //   "shape": [49,49],
    //   "data_offsets": [0,19208]
    // }
    IAllocator* allocator = get_allocator(DataOn::CPU);
    for (auto& m : doc.GetObject()) {
        if (m.value.GetObject().HasMember("dtype") == false) continue;
        ModelParam::SafeTensorInfo sti;
        int64_t total_number = 1;
        for (auto&v : m.value.GetObject()["shape"].GetArray()) {
            sti.shape.push_back(v.GetInt());
            total_number *= v.GetInt();
        }
        for (auto&v : m.value.GetObject()["data_offsets"].GetArray()) {
            if (v.IsInt64()) {
                sti.data_offset.push_back(v.GetInt64());
            } else if (v.IsInt()) {
                sti.data_offset.push_back(v.GetInt());
            }
        }
        // now load data into memory
        size_t byte_size = sti.data_offset[1]-sti.data_offset[0];
        if (m.value.GetObject()["dtype"].GetString() == std::string("I64")) {
            sti.dtype = TypeMeta::make<int64_t>();
            sti.data = allocator->alloc(byte_size);
            file.read(static_cast<char*>(sti.data), byte_size);
        } else if (m.value.GetObject()["dtype"].GetString() == std::string("F32")) {
            sti.dtype = TypeMeta::make<float>();
            sti.data = allocator->alloc(byte_size);
            file.read(static_cast<char*>(sti.data), byte_size);
        } else if (m.value.GetObject()["dtype"].GetString() == std::string("BF16")) {
            sti.dtype = TypeMeta::make<float>();
            void* _tmp_data = allocator->alloc(byte_size);
            file.read(static_cast<char*>(_tmp_data), byte_size);
            sti.data = allocator->alloc(byte_size*2);
            for (int i = 0; i < total_number; ++i) {
                uint16_t fp16_val = static_cast<uint16_t*>(_tmp_data)[i];
                static_cast<float*>(sti.data)[i] = bfloat16_to_float32(fp16_val);
            }
            allocator->free(_tmp_data);
        } else {
            MLOG(ERROR)<<"Unsupport dtype:"<<m.value.GetObject()["dtype"].GetString();
            return false;
        }
        
        
        callback(sti, param, m.name.GetString());
    }
    file.close();
    return true;
}

bool LModel::_backend_setup(GptParams& gpt_params, ExeContext& context) {
    if (gpt_params.backend == DataOn::CPU) {
        return true;
    } else if (gpt_params.backend == DataOn::GPU) {
        return m_graph->gpu_distribute();
    } else {
        MLOG(ERROR)<<"unsupport backend:"<<device_string(gpt_params.backend);
        return false;
    }
}

int32_t LModel::_sample(Tensor& logits, const std::vector<int32_t>& pre_ids, int offset, int size) {
    std::unordered_set<int32_t> ids_set(pre_ids.begin(), pre_ids.end());
    Tensor  logits_cpu = logits.cpu();
    int32_t ssize       = logits_cpu.dim_at(2);
    int32_t len        = logits_cpu.dim_at(1);
    float* scores = logits_cpu.ptr<float>((len-1)*ssize);
    for (auto id : ids_set) {
        float score = scores[id];
        scores[id]  = score < 0 ? score * _repetition_penalty : score / _repetition_penalty;
    }
    // argmax
    float max_score = scores[0];
    int32_t token_id = 0;
    for (int i = 1; i < ssize; i++) {
        float score = scores[i];
        if (score > max_score) {
            max_score = score;
            token_id  = i;
        }
    }
    return token_id;
}

bool LModel::generate(ExeContext& context, AIResult& result) {
    std::string prompt = m_tokenizer->apply_chat_template(context.prompt);
    std::vector<int32_t> tokens = m_tokenizer->encode(prompt);
    
    int32_t new_token_len = 0;
    int32_t cache_len = 0;
    while (new_token_len < context.max_text_len) {
        tensor_list otensors = _compute(context, tokens, cache_len);
        if (otensors.empty()) break;
        ++new_token_len;
        int32_t cur_token = _sample(otensors[0], tokens);
        if (m_tokenizer->is_stop(cur_token)) {
            break;
        }
        // cache_len += tokens.size();
        // tokens = {cur_token};
        tokens.push_back(cur_token);
        std::string token = m_tokenizer->decode(cur_token);
        std::cout<<std::unitbuf<<token;
    }
    std::cout << std::endl;
    // DUMP_TENSOR_TO_TXT(otensors[0].cpu(), "otensors");
    return true;
}

} // namespace mariana
