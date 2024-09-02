/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : device_type.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-17:20:32:41
 * Description:
 * 
 */

#include <core/device_type.h>

namespace mariana {


std::string device_string(const DataOn& device) {
    switch(device) {
    case DataOn::NONE:
        return "NONE";
    case DataOn::CPU:
        return "CPU";
    case DataOn::GPU:
        return "GPU";
    default:
        return "uinit";
    };
}

} // namespace mariana 
