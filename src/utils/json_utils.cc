/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : utils/json_utils.cc
 * Authors    : lqwang@SMT23090002
 * Create Time: 2025-03-03:16:03:27
 * Description:
 * 
 */

#include <fstream>

#include <utils/sys.h>
#include <utils/json_utils.h>
#include <utils/mariana_define.h>
#include <utils/rapidjson/document.h>

namespace mariana {

bool load_config(const char* config_json, AnyMap& any_map) {
    TRACE();
    if (false == file_exist(config_json)) {
        MLOG(ERROR)<<"config.json is not exist in:"<<config_json;
        return false;
    }
    std::ifstream in(config_json);
    if (!in.is_open()) {
        MLOG(ERROR)<<"Open "<<config_json<<" failed";
        return false;
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    in.close();
    std::string json_str(buffer.str());
    rapidjson::Document doc;
	doc.Parse(json_str.c_str());
    if (doc.HasParseError()) {
        MLOG(ERROR)<<"Parse json file failed:"<<config_json;
        return false;
    }
    
    std::function<AnyMap(const rapidjson::Document::Object&)> get_all_json_member = [&](const rapidjson::Document::Object& object)->AnyMap {
        AnyMap any_map;
        for (auto& obj : object) {
            JSON_ARRAY_HANDLE(obj.value, [&]()->void {
                if (obj.value.Empty()) return;
                JSON_DOUBLE_HANDLE(obj.value.GetArray()[0], [&]()->void {
                    std::vector<float> _tmp;
                    for (auto& item : obj.value.GetArray()) {
                        _tmp.push_back(item.GetDouble());
                    }
                    any_map[obj.name.GetString()] = ::absl::any(_tmp);
                }(), return);
                JSON_INT_HANDLE(obj.value.GetArray()[0], [&]()->void {
                    std::vector<int32_t> _tmp;
                    for (auto& item : obj.value.GetArray()) {
                        _tmp.push_back(item.GetInt());
                    }
                    any_map[obj.name.GetString()] = ::absl::any(_tmp);
                }(), return);
                JSON_STRING_HANDLE(obj.value.GetArray()[0], [&]()->void {
                    std::vector<std::string> _tmp;
                    for (auto& item : obj.value.GetArray()) {
                        _tmp.push_back(item.GetString());
                    }
                    any_map[obj.name.GetString()] = ::absl::any(_tmp);
                }(), return);
                JSON_OBJECT_HANDLE(obj.value.GetArray()[0], [&]()->void {
                    std::vector<AnyMap> _tmp;
                    for (auto& item : obj.value.GetArray()) {
                        AnyMap tmp_any_map;
                        _tmp.push_back(get_all_json_member(item.GetObject()));
                    }
                    any_map[obj.name.GetString()] = ::absl::any(_tmp);
                }(), return);
            }(), continue);
            JSON_INT_HANDLE(obj.value, any_map[obj.name.GetString()] = ::absl::any(obj.value.GetInt()), continue);
            JSON_BOOL_HANDLE(obj.value, any_map[obj.name.GetString()] = ::absl::any(obj.value.GetBool()), continue);
            JSON_DOUBLE_HANDLE(obj.value, any_map[obj.name.GetString()] =
                               ::absl::any(static_cast<float>(obj.value.GetDouble())), continue);
            JSON_STRING_HANDLE(obj.value, any_map[obj.name.GetString()] = ::absl::any(std::string(obj.value.GetString())), continue);
            
            JSON_OBJECT_HANDLE(obj.value, any_map[obj.name.GetString()] = ::absl::any(get_all_json_member(obj.value.GetObject())), continue);
        }
        return any_map;
    };
    any_map = get_all_json_member(doc.GetObject());
    return true;
}

} // namespace mariana
