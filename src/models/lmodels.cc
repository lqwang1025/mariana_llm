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

#include <utils/sys.h>

#include <models/lmodels.h>

#include <core/graph.h>
#include <core/impl/allocator.h>
#include <core/device_type.h>

#include <utils/mariana_define.h>
#include <utils/dtype_utils.h>
#include <utils/rapidjson/document.h>

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

} // namespace mariana
