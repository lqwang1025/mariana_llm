/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/act_route.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-27:15:43:21
 * Description:
 * 
 */

#include <ops/backend/cpu/act_route.h>
#include <ops/backend/cpu/gelu.h>
#include <utils/mariana_define.h>

namespace mariana {

float act_route(OpCategory opcate, float x) {
    switch(opcate) {
    case OpCategory::GELU:
        return gelu_single(x);
    case OpCategory::RELU:
        return relu_single(x);
    default:
        return x;
    };
}

} // namespace mariana
