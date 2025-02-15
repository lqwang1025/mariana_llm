/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : utils/dtype_utils.h
 * Authors    : lqwang@pandora
 * Create Time: 2025-02-15:19:54:39
 * Description:
 *
 */

#ifndef __DTYPE_UTILS_H__
#define __DTYPE_UTILS_H__

namespace mariana {

inline unsigned short float32_to_bfloat16(float value) {
    // 16 : 16
    union {
        unsigned int u;
        float f;
    } tmp;
    tmp.f = value;
    return tmp.u >> 16;
}

// convert brain half to float
inline float bfloat16_to_float32(unsigned short value) {
    // 16 : 16
    union {
        unsigned int u;
        float f;
    } tmp;
    tmp.u = value << 16;
    return tmp.f;
}

} // namespace mariana

#endif /* __DTYPE_UTILS_H__ */

