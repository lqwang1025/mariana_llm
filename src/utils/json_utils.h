/*
 *        (C) COPYRIGHT Ingenic Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : utils/json_utils.h
 * Authors    : lqwang@SMT23090002
 * Create Time: 2025-03-03:16:01:24
 * Description:
 *
 */

#ifndef __UTILS_JSON_UTILS_H__
#define __UTILS_JSON_UTILS_H__

#include <unordered_map>

#include <absl/types/any.h>

namespace mariana {

using AnyMap = std::unordered_map<std::string, ::absl::any>;

bool load_config(const char* config_json, AnyMap& any_map);

} // namespace mariana

#endif /* __UTILS_JSON_UTILS_H__ */

