/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/x86_x64/avx/avx_common.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-06-24:00:45:20
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_X86_X64_AVX_AVX_COMMON_H__
#define __OPS_BACKEND_CPU_X86_X64_AVX_AVX_COMMON_H__

#include <immintrin.h>

namespace mariana {

float __avx_sum8(__m256 x);

float __avx_vec_mul_8_fp32(const float* a, const float* b);

void __avx_softmax_fp32(const float* src, float* dst, size_t size);

void _avx_GELU_fp32(float *dst, const float *src, size_t size, float* parameters);

void _avx_norm_fp32(float *dst, const float *src, const float *gamma, const float *beta, float epsilon, size_t size, bool RMSNorm);

} // namespace mariana

#endif /* __OPS_BACKEND_CPU_X86_X64_AVX_AVX_COMMON_H__ */

