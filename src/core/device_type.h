/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/device_type.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-15:08:51:11
 * Description:
 *
 */

#ifndef __DEVICE_TYPE_H__
#define __DEVICE_TYPE_H__

#include <string>
#include <cstdint>

#include <mariana_llm/mariana_llm_impl.h>

namespace mariana {

std::string device_string(const DataOn& device);

} // namespace mariana 

#endif /* __DEVICE_TYPE_H__ */

