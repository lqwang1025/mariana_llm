/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : avx_common.cc
 * Authors    : lqwang@pandora
 * Create Time: 2024-06-24:00:46:42
 * Description:
 * 
 */

#include <cmath>

#include <ops/backend/cpu/x86_x64/avx/avx_common.h>

namespace mariana {

float __avx_sum8(__m256 x) {
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

float __avx_vec_mul_8_fp32(const float* a, const float* b) {
    __m256 A = _mm256_loadu_ps(a);
    __m256 B = _mm256_loadu_ps(b);
    __m256 C = _mm256_mul_ps(A, B);
    return  __avx_sum8(C);
}

void _avx_GELU_fp32(float *dst, const float *src, size_t size, float* parameters) {
    // parameters[8] = {0.044715f, 0.79788458f, 378.f, 17325.f, 135135.f, 28.f, 3150.f, 62370.f};
    auto var1 = _mm256_set1_ps(parameters[0]);
    auto var2 = _mm256_set1_ps(parameters[1]);
    auto var3 = _mm256_set1_ps(parameters[2]);
    auto var4 = _mm256_set1_ps(parameters[3]);
    auto var5 = _mm256_set1_ps(parameters[4]);
    auto var6 = _mm256_set1_ps(parameters[5]);
    auto var7 = _mm256_set1_ps(parameters[6]);
    auto var8 = _mm256_set1_ps(parameters[7]);
    auto var9 = _mm256_set1_ps(parameters[4]);
    auto var10 = _mm256_set1_ps(0.5);
    auto varOne = _mm256_set1_ps(1.f);
    auto varNegOne = _mm256_set1_ps(-1.f);
    auto clamp_min = _mm256_set1_ps(-5.0f);
    auto clamp_max = _mm256_set1_ps(5.0f);
    for (size_t i = 0; i < size; i++) {
        auto x = _mm256_loadu_ps(src + i * 8);
        auto y = _mm256_mul_ps(x, x);
        y = _mm256_mul_ps(y, x);
        y = _mm256_mul_ps(y, var1);
        y = _mm256_add_ps(y, x);
        y = _mm256_mul_ps(y, var2);
        y = _mm256_max_ps(y, clamp_min);
        y = _mm256_min_ps(y, clamp_max);
        // y = tanh(y)
        {
            auto y2 = _mm256_mul_ps(y, y);
            auto w = _mm256_add_ps(y2, var3);
            w = _mm256_mul_ps(w, y2);
            w = _mm256_add_ps(w, var4);
            w = _mm256_mul_ps(w, y2);
            w = _mm256_add_ps(w, var5);
            w = _mm256_mul_ps(w, y);
            auto z = _mm256_mul_ps(y2, var6);
            z = _mm256_add_ps(z, var7);
            z = _mm256_mul_ps(z, y2);
            z = _mm256_add_ps(z, var8);
            z = _mm256_mul_ps(z, y2);
            z = _mm256_add_ps(z, var9);
            z = _mm256_div_ps(w, z);
            z = _mm256_max_ps(z, varNegOne);
            y = _mm256_min_ps(z, varOne);
        }
        y = _mm256_add_ps(y, varOne);
        y = _mm256_mul_ps(y, x);
        y = _mm256_mul_ps(y, var10);
        _mm256_storeu_ps(dst + i * 8, y);
    }
}

__m256 exp256_ps(__m256 x) {
/* Modified code. The original code is here: https://github.com/reyoung/avx_mathfun

   AVX implementation of exp
   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/
   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)
   This software is provided 'as-is', without any express or implied
   warranty.  In no event will the authors be held liable for any damages
   arising from the use of this software.
   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:
   1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
   2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
   3. This notice may not be removed or altered from any source distribution.
   (this is the zlib license)
*/
/* 
   To increase the compatibility across different compilers the original code is
   converted to plain AVX2 intrinsics code without ingenious macro's,
   gcc style alignment attributes etc. The modified code requires AVX2
*/
    __m256   exp_hi        = _mm256_set1_ps(88.3762626647949f);
    __m256   exp_lo        = _mm256_set1_ps(-88.3762626647949f);

    __m256   cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341);
    __m256   cephes_exp_C1 = _mm256_set1_ps(0.693359375);
    __m256   cephes_exp_C2 = _mm256_set1_ps(-2.12194440e-4);

    __m256   cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4);
    __m256   cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3);
    __m256   cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3);
    __m256   cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2);
    __m256   cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1);
    __m256   cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1);
    __m256   tmp           = _mm256_setzero_ps(), fx;
    __m256i  imm0;
    __m256   one           = _mm256_set1_ps(1.0f);

    x     = _mm256_min_ps(x, exp_hi);
    x     = _mm256_max_ps(x, exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx    = _mm256_mul_ps(x, cephes_LOG2EF);
    fx    = _mm256_add_ps(fx, _mm256_set1_ps(0.5f));
    tmp   = _mm256_floor_ps(fx);
    __m256  mask  = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);    
    mask  = _mm256_and_ps(mask, one);
    fx    = _mm256_sub_ps(tmp, mask);
    tmp   = _mm256_mul_ps(fx, cephes_exp_C1);
    __m256  z     = _mm256_mul_ps(fx, cephes_exp_C2);
    x     = _mm256_sub_ps(x, tmp);
    x     = _mm256_sub_ps(x, z);
    z     = _mm256_mul_ps(x,x);

    __m256  y     = cephes_exp_p0;
    y     = _mm256_mul_ps(y, x);
    y     = _mm256_add_ps(y, cephes_exp_p1);
    y     = _mm256_mul_ps(y, x);
    y     = _mm256_add_ps(y, cephes_exp_p2);
    y     = _mm256_mul_ps(y, x);
    y     = _mm256_add_ps(y, cephes_exp_p3);
    y     = _mm256_mul_ps(y, x);
    y     = _mm256_add_ps(y, cephes_exp_p4);
    y     = _mm256_mul_ps(y, x);
    y     = _mm256_add_ps(y, cephes_exp_p5);
    y     = _mm256_mul_ps(y, z);
    y     = _mm256_add_ps(y, x);
    y     = _mm256_add_ps(y, one);

    /* build 2^n */
    imm0  = _mm256_cvttps_epi32(fx);
    imm0  = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
    imm0  = _mm256_slli_epi32(imm0, 23);
    __m256  pow2n = _mm256_castsi256_ps(imm0);
    y     = _mm256_mul_ps(y, pow2n);
    return y;
}

void __avx_softmax_fp32(const float* source, float* dest, size_t size) {
    float tmpfloat8[8];
    int count  = size / 8;
    int remain = count * 8;
    // step 1: get maxValue
    float maxValue = source[0];
    if (count > 0) {
        auto maxVal = _mm256_loadu_ps(source);
        for (int i = 1; i < count; i++) {
            maxVal = _mm256_max_ps(maxVal, _mm256_loadu_ps(source + i * 8));
        }
        _mm256_storeu_ps(tmpfloat8, maxVal);
        maxValue = tmpfloat8[0] > tmpfloat8[1] ? tmpfloat8[0] : tmpfloat8[1];
        for (int i = 2; i < 8; i++) {
            maxValue = maxValue > tmpfloat8[i] ? maxValue : tmpfloat8[i];
        }
    }
    for (size_t i = remain; i < size; i++) {
        maxValue = maxValue > source[i] ? maxValue : source[i];
    }

    // step 2: get exp(x - maxValue) and sum(exp(x - maxValue))
    float sumValue = 0.f;
    if (count > 0) {
        auto sumVal = _mm256_set1_ps(0.f);
        auto p0    = _mm256_set1_ps(0.6931471805599453);
        auto p1    = _mm256_set1_ps(1.4426950408889634);
        auto p2    = _mm256_set1_ps(1.f);
        auto p3    = _mm256_set1_ps(1.f);
        auto p4    = _mm256_set1_ps(0.5);
        auto p5    = _mm256_set1_ps(0.1666666666666666);
        auto p6    = _mm256_set1_ps(0.041666666666666664);
        auto p7    = _mm256_set1_ps(0.008333333333333333);
        auto xMax  = _mm256_set1_ps(87);
        auto xMin  = _mm256_set1_ps(-87);
        auto basic = _mm256_set1_epi32(1 << 23);
        auto temp127 = _mm256_set1_epi32(127);
        for (int i = 0; i < count; ++i) {
            auto x            = _mm256_sub_ps(_mm256_loadu_ps(source + i * 8), _mm256_set1_ps(maxValue));
            x                 = _mm256_max_ps(x, xMin);
            x                 = _mm256_min_ps(x, xMax);
            auto div          = _mm256_mul_ps(x, p1);
            auto divInt       = _mm256_cvtps_epi32(div);
            div               = _mm256_cvtepi32_ps(divInt);
            auto div2         = _mm256_add_epi32(divInt, temp127);
            div2 = _mm256_mullo_epi32(div2, basic);
            auto expBasic  = _mm256_castsi256_ps(div2);
            auto xReamin   = _mm256_sub_ps(x, _mm256_mul_ps(div, p0));
            auto t         = xReamin;
            auto c0        = _mm256_mul_ps(p7, t);
            auto c1        = _mm256_add_ps(c0, p6);
            auto c2        = _mm256_mul_ps(c1, t);
            auto c3        = _mm256_add_ps(c2, p5);
            auto c4        = _mm256_mul_ps(c3, t);
            auto c5        = _mm256_add_ps(c4, p4);
            auto c6        = _mm256_mul_ps(c5, t);
            auto c7        = _mm256_add_ps(c6, p3);
            auto c8        = _mm256_mul_ps(c7, t);
            auto c9        = _mm256_add_ps(c8, p2);
            auto expRemain = c9;
            auto expRes    = _mm256_mul_ps(expBasic, expRemain);
            sumVal         = _mm256_add_ps(expRes, sumVal);
            _mm256_storeu_ps(dest + 8 * i, expRes);
        }
        _mm256_storeu_ps(tmpfloat8, sumVal);
        for (int i = 0; i < 8; i++) {
            sumValue += tmpfloat8[i];
        }
    }
    auto param = 0.6931471805599453;
    float xLimit = 87;
    for (size_t i = remain; i < size; i++) {
        auto x         = source[i] - maxValue;
        x = x > -xLimit ? x : -xLimit;
        x = x < xLimit ? x : xLimit;

        int div        = (x / param);
        int div2       = (div + 127) << 23;
        auto xReamin   = x - div * param;
        float expBasic = *reinterpret_cast<float*>(&div2);

        auto t         = xReamin;
        auto expRemain = ((((1.0f / 120 * t + 1.0f / 24) * t + 1.0f / 6) * t + 0.5f) * t + 1.0f) * t + 1.0f;
        dest[i]  = expBasic * expRemain;
        sumValue += dest[i];
    }
    // step 3: get x / sum and store
    for (int i = 0; i < count; ++i) {
        // using  1 / ((1 / x) * sum) instead x * (1 / sum) or x / sum for some bugs in intel cpu
        auto x = _mm256_rcp_ps(_mm256_loadu_ps(dest + 8 * i));
        auto y = _mm256_set1_ps(sumValue);
        auto z = _mm256_rcp_ps(_mm256_mul_ps(x, y));
        _mm256_storeu_ps(dest + 8 * i, z);
    }
    sumValue = 1.f / sumValue;
    for (size_t i = remain; i < size; i++) {
        dest[i] *= sumValue;
    }
}

void _avx_norm_fp32(float *dst, const float *src, const float *gamma, const float *beta, float epsilon, size_t size, bool RMSNorm) {
    float tmpfloat8[8];
    int count  = static_cast<int32_t>(size / 8);
    int remain = count * 8;
    // step 1: get sum
    float mean = 0;
    if(!RMSNorm){
        float sum = 0.f;
        if (count > 0) {
            auto sumVal = _mm256_set1_ps(0.f);
            for (int i = 0; i < count; i++) {
                sumVal = _mm256_add_ps(sumVal, _mm256_loadu_ps(src + i * 8));
            }
            _mm256_storeu_ps(tmpfloat8, sumVal);
            for (int i = 0; i < 8; i++) {
                sum += tmpfloat8[i];
            }
        }
        for (size_t i = remain; i < size; i++) {
            sum += src[i];
        }
        mean = sum / size;
    }
    // step 2: get square_sum
    float square_sum = 0.f;
    auto meanVal = _mm256_set1_ps(mean);
    if (count > 0) {
        auto sumVal = _mm256_set1_ps(0.f);
        for (int i = 0; i < count; i++) {
            auto x = _mm256_sub_ps(_mm256_loadu_ps(src + i * 8), meanVal);
            sumVal = _mm256_add_ps(sumVal, _mm256_mul_ps(x, x));
        }
        _mm256_storeu_ps(tmpfloat8, sumVal);
        for (int i = 0; i < 8; i++) {
            square_sum += tmpfloat8[i];
        }
    }
    for (size_t i = remain; i < size; i++) {
        float x = (src[i] - mean);
        square_sum += x * x;
    }
    // step 3: get result
    float variable = square_sum / size;
    variable = 1.f / sqrt(variable + epsilon);
    auto variableVal = _mm256_set1_ps(variable);
    if (gamma && beta) {
        for (int i = 0; i < count; i++) {
            auto x = _mm256_sub_ps(_mm256_loadu_ps(src + i * 8), meanVal);
            auto g = _mm256_loadu_ps(gamma + i * 8);
            auto b = _mm256_loadu_ps(beta + i * 8);
            auto y = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(x, g), variableVal), b);
            _mm256_storeu_ps(dst + i * 8, y);
        }
        for (size_t i = remain; i < size; i++) {
            dst[i] = (src[i] - mean) * gamma[i] * variable + beta[i] ;
        }
    } else {
        for (int i = 0; i < count; i++) {
            auto x = _mm256_sub_ps(_mm256_loadu_ps(src + i * 8), meanVal);
            auto y = _mm256_mul_ps(x, variableVal);
            _mm256_storeu_ps(dst + i * 8, y);
        }
        for (size_t i = remain; i < size; i++) {
            dst[i] = (src[i] - mean) * variable;
        }
    }
}

} // namespace mariana
