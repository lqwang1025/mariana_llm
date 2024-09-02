/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/matmul.h
 * Authors    : lqwang@pandora
 * Create Time: 2024-06-23:19:23:44
 * Description:
 * 
 */

#include <utils/mariana_define.h>

#include <ops/backend/cpu/matmul.h>
#include <ops/backend/cpu/act_route.h>
#include <ops/backend/cpu/x86_x64/avx/avx_funcs.h>

namespace mariana {

[[maybe_unused]] static void _naive_matmul_fp32(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& out, float alpha, float beta, OpCategory act_cate) {
    //todo : process with activation function
    const uint32_t noc = weight.dim_at(0); // number of row
    const uint32_t nok = weight.dim_at(1); // number of row
    for (uint32_t row = sched_param.this_thread_begin_index(); row < sched_param.this_thread_end_index(); ++row) {
        for (uint32_t col = 0; col < noc; ++col) {
            float sum = 0.f;
            for (uint32_t k = 0; k < nok; ++k) {
                sum += input.data_at<float>(row*nok+k)*weight.data_at<float>(col*nok+k);
            }
            float _bias = 0.f;
            if (noc != bias.total_size()) {
                _bias = bias.data_at<float>(row);
            } else {
                _bias = bias.data_at<float>(col);
            }
            *out.unsafe_ptr<float>(row*noc+col) = act_route(act_cate, alpha*sum + beta*_bias);
        }
    }
}

void matmul(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& out, float alpha, float beta, OpCategory act_cate) {
    if (bias.total_size() != 0) {
        _avx2_gemm_fp32(sched_param, input, weight, bias, out, alpha, beta, act_cate);
        // _naive_matmul_fp32(sched_param, input, weight, bias, out, act_cate);
    } else {
        _avx2_gemm_no_bias_fp32(sched_param, input, weight, out, alpha, act_cate);
    }
}

} // namespace mariana

