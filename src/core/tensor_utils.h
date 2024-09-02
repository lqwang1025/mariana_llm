/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : core/tensor_utils.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-23:08:15:16
 * Description:
 *
 */

#ifndef __CORE_TENSOR_UTILS_H__
#define __CORE_TENSOR_UTILS_H__

#include <string>

namespace mariana {

class Tensor;

void to_bin(const Tensor& tensor, const std::string& name);

void to_txt(const Tensor& tensor, const std::string& name);

#define DUMP_TENSOR_TO_BIN(t, name)                                     \
    do {                                                                \
        static uint32_t count = 0;                                      \
        to_bin(t, std::string(name)+"_"+std::to_string(count++)+".bin"); \
    } while (false)

#define DUMP_TENSOR_TO_TXT(t, name)                                     \
    do {                                                                \
        static uint32_t count = 0;                                      \
        to_txt(t, std::string(name)+"_"+std::to_string(count++)+".txt"); \
    } while (false)


} // namespace mariana

#endif /* __CORE_TENSOR_UTILS_H__ */

