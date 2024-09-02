/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/x86_x64/avx/avx_funcs.cc
 * Authors    : lqwang@pandora
 * Create Time: 2024-06-23:22:21:22
 * Description:
 * 
 */

#include <cmath>
#include <cfloat>

#include <core/tensor.h>
#include <ops/backend/cpu/x86_x64/avx/avx_funcs.h>

#include <utils/mariana_define.h>

#include <ops/backend/cpu/act_route.h>
#include <ops/backend/cpu/x86_x64/avx/avx_common.h>

namespace mariana {

void _avx2_gemm_fp32(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& out, float alpha, float beta, OpCategory act_cate) {
    const uint32_t noc = weight.dim_at(0); // number of row
    const uint32_t nok = weight.dim_at(1); // number of row
    uint32_t _8_chunk  = nok/8;
    for (uint32_t row = sched_param.this_thread_begin_index(); row < sched_param.this_thread_end_index(); ++row) {
        for (uint32_t col = 0; col < noc; ++col) {
            float sum = 0.f;
            for (uint32_t k = 0; k < _8_chunk; ++k) {
                const float* a = input.unsafe_ptr<float>(row*nok+k*8);
                const float* b = weight.unsafe_ptr<float>(col*nok+k*8);
                sum += __avx_vec_mul_8_fp32(a, b);
            }
            for (uint32_t k = _8_chunk*8; k < nok; ++k) {
                sum += input.data_at<float>(row*nok+k)*weight.data_at<float>(col*nok+k);
            }
            float _bias = 0.f;
            if (noc != bias.total_size()) {
                _bias = bias.data_at<float>(row);
            } else {
                _bias = bias.data_at<float>(col);
            }
            *out.unsafe_ptr<float>(row*noc+col) = act_route(act_cate, alpha*sum+beta*_bias);
        }
    }
}

void _avx2_gemm_no_bias_fp32(SchedParam sched_param, const Tensor& input, const Tensor& weight, Tensor& out, float alpha, OpCategory act_cate) {
    const uint32_t noc = weight.dim_at(0); // number of row
    const uint32_t nok = weight.dim_at(1); // number of row
    uint32_t _8_chunk  = nok/8;
    for (uint32_t row = sched_param.this_thread_begin_index(); row < sched_param.this_thread_end_index(); ++row) {
        for (uint32_t col = 0; col < noc; ++col) {
            float sum = 0.f;
            for (uint32_t k = 0; k < _8_chunk; ++k) {
                const float* a = input.unsafe_ptr<float>(row*nok+k*8);
                const float* b = weight.unsafe_ptr<float>(col*nok+k*8);
                sum += __avx_vec_mul_8_fp32(a, b);
            }
            for (uint32_t k = _8_chunk*8; k < nok; ++k) {
                sum += input.data_at<float>(row*nok+k)*weight.data_at<float>(col*nok+k);
            }
            *out.unsafe_ptr<float>(row*noc+col) = act_route(act_cate, alpha*sum);
        }
    }
}

void _avx2_mhs_mask_attention_batch_split_fp32(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& mask, Tensor& out, int32_t n_head, int32_t head_size) {
    float scale = 1.f/sqrtf(head_size);
    int32_t B = Q.dim_at(0); // batch
    uint32_t T = Q.dim_at(1); // 52
    float _qk_dot[T] = {0.f};
    for (uint32_t row = sched_param.this_thread_begin_index(); row < sched_param.this_thread_end_index(); ++row) {
        uint32_t b = row/T;
        uint32_t t = row%T;
        for (int32_t h = 0; h < n_head; ++h) {
            //1. Q@V
            float maxval = -FLT_MAX;
            for (uint32_t t2 = 0; t2 < T; ++t2) { // 52
                float val = 0.f;
                uint32_t _8_chunk    = head_size/8;
                for (uint32_t i = 0; i < _8_chunk; ++i) { // 64
                    const float* A = Q.unsafe_ptr<float>(b*Q.stride_at(0)+t*Q.stride_at(1)+h*head_size+i*8);
                    
                    const float* B = K.unsafe_ptr<float>(b*K.stride_at(0)+t2*K.stride_at(1)+h*head_size+i*8);
                    val += __avx_vec_mul_8_fp32(A, B);
                }
                for (int32_t i = 8*_8_chunk; i < head_size; ++i) {
                    float q = Q.data_at<float>(b*Q.stride_at(0)+t*Q.stride_at(1)+h*head_size+i);
                    float k = K.data_at<float>(b*K.stride_at(0)+t2*K.stride_at(1)+h*head_size+i);
                    val += q*k;
                }
                uint32_t mask_id = t*mask.stride_at(2)+t2;
                if (mask.dim_at(0) == B) {
                    mask_id += b*mask.stride_at(0);
                }
                if (mask.dim_at(1) == n_head) {
                    mask_id += h*mask.stride_at(1);
                }
                _qk_dot[t2] = val*scale+mask.data_at<float>(mask_id);
                if (_qk_dot[t2] > maxval) {
                    maxval = _qk_dot[t2];
                }
            }
            
            // 2. attention scores softmax
            float _softmaxed_qk[T] = {0.f};
            __avx_softmax_fp32(_qk_dot, _softmaxed_qk, T);
            
            // 3. attention_scores @ V
            for (int32_t i = 0; i < head_size; ++i) {
                *out.unsafe_ptr<float>(b*out.stride_at(0)+t*out.stride_at(1)+h*head_size+i) = 0;
            }
            
            for (uint32_t t2 = 0; t2 < T; ++t2) {
                for (int32_t i = 0; i < head_size; ++i) {
                    float value = V.data_at<float>(b*V.stride_at(0)+t2*V.stride_at(1)+h*head_size+i);
                    *out.unsafe_ptr<float>(b*out.stride_at(0)+t*out.stride_at(1)+h*head_size+i) += value*_softmaxed_qk[t2];
                }
            }
        } // n_head
    } // row
}

void _avx2_mhs_attention_batch_split_fp32(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& out, int32_t n_head, int32_t head_size) {
    float scale = 1.f/sqrtf(head_size);
    int32_t B = Q.dim_at(0); // batch
    uint32_t QT = Q.dim_at(1); // 52
    uint32_t VT = V.dim_at(1); // 52
    uint32_t T  = K.dim_at(1); // 52
    float _qk_dot[T] = {0.f};
    for (uint32_t row = sched_param.this_thread_begin_index(); row < sched_param.this_thread_end_index(); ++row) {
        uint32_t b = row/QT;
        uint32_t t = row%QT;
        for (int32_t h = 0; h < n_head; ++h) {
            //1. Q@V
            float maxval = -FLT_MAX;
            for (uint32_t t2 = 0; t2 < T; ++t2) { // 52
                float val = 0.f;
                uint32_t _8_chunk    = head_size/8;
                for (uint32_t i = 0; i < _8_chunk; ++i) { // 64
                    const float* A = Q.unsafe_ptr<float>(b*Q.stride_at(0)+t*Q.stride_at(1)+h*head_size+i*8);
                    
                    const float* B = K.unsafe_ptr<float>(b*K.stride_at(0)+t2*K.stride_at(1)+h*head_size+i*8);
                    val += __avx_vec_mul_8_fp32(A, B);
                }
                for (int32_t i = 8*_8_chunk; i < head_size; ++i) {
                    float q = Q.data_at<float>(b*Q.stride_at(0)+t*Q.stride_at(1)+h*head_size+i);
                    float k = K.data_at<float>(b*K.stride_at(0)+t2*K.stride_at(1)+h*head_size+i);
                    val += q*k;
                }
                
                _qk_dot[t2] = val*scale;
                if (_qk_dot[t2] > maxval) {
                    maxval = _qk_dot[t2];
                }
            }
            
            // 2. attention scores softmax
            float _softmaxed_qk[T] = {0.f};
            __avx_softmax_fp32(_qk_dot, _softmaxed_qk, T);
            
            // 3. attention_scores @ V
            for (int32_t i = 0; i < head_size; ++i) {
                *out.unsafe_ptr<float>(b*out.stride_at(0)+t*out.stride_at(1)+h*head_size+i) = 0;
            }
            
            for (uint32_t t2 = 0; t2 < VT; ++t2) {
                for (int32_t i = 0; i < head_size; ++i) {
                    float value = V.data_at<float>(b*V.stride_at(0)+t2*V.stride_at(1)+h*head_size+i);
                    *out.unsafe_ptr<float>(b*out.stride_at(0)+t*out.stride_at(1)+h*head_size+i) += value*_softmaxed_qk[t2];
                }
            }
        } // n_head
    } // row
}

void _avx2_mhs_swin_mask_attention_batch_split_fp32(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& mask, const Tensor& attn_mask, Tensor& out, int32_t n_head, int32_t head_size) {
        float scale = 1.f/sqrtf(head_size);
    int32_t B = Q.dim_at(0); // batch
    uint32_t T = Q.dim_at(1); // 52
    float _qk_dot[T] = {0.f};
    for (uint32_t row = sched_param.this_thread_begin_index(); row < sched_param.this_thread_end_index(); ++row) {
        uint32_t b = row/T;
        uint32_t t = row%T;
        for (int32_t h = 0; h < n_head; ++h) {
            //1. Q@V
            float maxval = -FLT_MAX;
            for (uint32_t t2 = 0; t2 < T; ++t2) { // 52
                float val = 0.f;
                uint32_t _8_chunk    = head_size/8;
                for (uint32_t i = 0; i < _8_chunk; ++i) { // 64
                    const float* A = Q.unsafe_ptr<float>(b*Q.stride_at(0)+t*Q.stride_at(1)+h*head_size+i*8);
                    
                    const float* B = K.unsafe_ptr<float>(b*K.stride_at(0)+t2*K.stride_at(1)+h*head_size+i*8);
                    val += __avx_vec_mul_8_fp32(A, B);
                }
                for (int32_t i = 8*_8_chunk; i < head_size; ++i) {
                    float q = Q.data_at<float>(b*Q.stride_at(0)+t*Q.stride_at(1)+h*head_size+i);
                    float k = K.data_at<float>(b*K.stride_at(0)+t2*K.stride_at(1)+h*head_size+i);
                    val += q*k;
                }
                uint32_t mask_id = t*mask.stride_at(2)+t2;
                if (mask.dim_at(0) == B) {
                    mask_id += b*mask.stride_at(0);
                }
                if (mask.dim_at(1) == n_head) {
                    mask_id += h*mask.stride_at(1);
                }
                uint32_t att_mask_id = b*attn_mask.stride_at(0)+t*attn_mask.stride_at(2)+t2;
                _qk_dot[t2] = val*scale+mask.data_at<float>(mask_id)+attn_mask.data_at<float>(att_mask_id);
                if (_qk_dot[t2] > maxval) {
                    maxval = _qk_dot[t2];
                }
            }
            
            // 2. attention scores softmax
            float _softmaxed_qk[T] = {0.f};
            __avx_softmax_fp32(_qk_dot, _softmaxed_qk, T);
            
            // 3. attention_scores @ V
            for (int32_t i = 0; i < head_size; ++i) {
                *out.unsafe_ptr<float>(b*out.stride_at(0)+t*out.stride_at(1)+h*head_size+i) = 0;
            }
            
            for (uint32_t t2 = 0; t2 < T; ++t2) {
                for (int32_t i = 0; i < head_size; ++i) {
                    float value = V.data_at<float>(b*V.stride_at(0)+t2*V.stride_at(1)+h*head_size+i);
                    *out.unsafe_ptr<float>(b*out.stride_at(0)+t*out.stride_at(1)+h*head_size+i) += value*_softmaxed_qk[t2];
                }
            }
        } // n_head
    } // row
}

void _avx2_bimhs_grdino_qk_dot_batch_split(SchedParam sched_param, const Tensor& Q, const Tensor& K, Tensor& attn_weights, int32_t n_head, int32_t head_size, std::atomic<float>& max_val) {
    uint32_t T = Q.dim_at(1);
    uint32_t key_t = K.dim_at(1); // 52
    for (uint32_t row = sched_param.this_thread_begin_index(); row < sched_param.this_thread_end_index(); ++row) {
        uint32_t b = row/T;
        uint32_t t = row%T; // 20906
        for (int32_t h = 0; h < n_head; ++h) { // 4
            //1. Q@V
            for (uint32_t t2 = 0; t2 < key_t; ++t2) { // 52
                float val = 0.f;
                uint32_t _8_chunk    = head_size/8;
                for (uint32_t i = 0; i < _8_chunk; ++i) { // 64
                    const float* A = Q.unsafe_ptr<float>(b*Q.stride_at(0)+t*Q.stride_at(1)+h*head_size+i*8);
                    const float* B = K.unsafe_ptr<float>(b*K.stride_at(0)+t2*K.stride_at(1)+h*head_size+i*8);
                    val += __avx_vec_mul_8_fp32(A, B);
                }
                for (int32_t i = 8*_8_chunk; i < head_size; ++i) {
                    float q = Q.data_at<float>(b*Q.stride_at(0)+t*Q.stride_at(1)+h*head_size+i);
                    float k = K.data_at<float>(b*K.stride_at(0)+t2*K.stride_at(1)+h*head_size+i);
                    val += q*k;
                }
                if (val > max_val.load()) {
                    max_val = val;
                }
                *attn_weights.unsafe_ptr<float>(h*attn_weights.stride_at(0)+t*attn_weights.stride_at(1)+t2) = val;
            }
        } // n_head
    }
}

void _avx2_bimhs_grdino_attn_clamp_bt_split(SchedParam sched_param, Tensor& attn_weights, Tensor& text_attn_weights, const float& max_val, float clamp_min_val, float clamp_max_val) {
    uint32_t C = (uint32_t)attn_weights.dim_at(2);
    uint32_t T = (uint32_t)attn_weights.dim_at(1);
    std::vector<float> row_max_vals;
    row_max_vals.resize(C, -FLT_MAX);
    
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        uint32_t b = i;
        
        for (uint32_t t = 0; t < T; ++t) { // 20906
            for (uint32_t c = 0; c < C; ++c) { // 52
                uint32_t index = b*attn_weights.stride_at(0)+t*attn_weights.stride_at(1)+c;
                float attn_weight = *attn_weights.unsafe_ptr<float>(index);
                attn_weight -= max_val;
                attn_weight = MIN(attn_weight, clamp_max_val);
                attn_weight = MAX(attn_weight, clamp_min_val);
                *attn_weights.unsafe_ptr<float>(index) = attn_weight;
                if (attn_weight > row_max_vals[c]) {
                    row_max_vals[c] = attn_weight;
                }
            }
        }
        
        for (uint32_t c = 0; c < C; ++c) { // 52
            for (uint32_t t = 0; t < T; ++t) { // 20906
                uint32_t index = b*attn_weights.stride_at(0)+t*attn_weights.stride_at(1)+c;
                float attn_weight = *attn_weights.unsafe_ptr<float>(index);
                attn_weight -= row_max_vals[c];
                attn_weight = MIN(attn_weight, clamp_max_val);
                attn_weight = MAX(attn_weight, clamp_min_val);
                *text_attn_weights.unsafe_ptr<float>(b*text_attn_weights.stride_at(0)+c*text_attn_weights.stride_at(1)+t) = attn_weight;
            }
            // text_attn_weight softmax
            uint32_t index = b*text_attn_weights.stride_at(0)+c*text_attn_weights.stride_at(1);
            __avx_softmax_fp32(text_attn_weights.unsafe_ptr<float>(index), text_attn_weights.unsafe_ptr<float>(index), T);
        }

        // vision_attn_weight softmax
        for (uint32_t t = 0; t < T; ++t) { // 20906
            uint32_t index = b*attn_weights.stride_at(0)+t*attn_weights.stride_at(1);
            __avx_softmax_fp32(attn_weights.unsafe_ptr<float>(index), attn_weights.unsafe_ptr<float>(index), C);
        }
    }
}

void _avx2_bimhs_grdino_attn_v_dot_batch_split(SchedParam sched_param, const Tensor& attn_score, const Tensor& value, Tensor& attn_output, int32_t n_head, int32_t head_size) {
    // input1: 4 20906 52 input2: [1, 4 256 52] out: 4 20906 256
    int32_t T  = attn_score.dim_at(1); // 20906
    int32_t K  = attn_score.dim_at(2); // 52
        
    for (uint32_t row = sched_param.this_thread_begin_index(); row < sched_param.this_thread_end_index(); ++row) {
        uint32_t h = row/T;
        uint32_t t = row%T;
        for (uint32_t hs = 0; hs < (uint32_t)head_size; ++hs) { // 256
            float sum = 0.f;
            uint32_t _8_chunk    = K/8;
            for (uint32_t k = 0; k < _8_chunk; ++k) { // 52
                const float* A = attn_score.unsafe_ptr<float>(h*attn_score.stride_at(0)+t*attn_score.stride_at(1)+k*8);
                const float* B = value.unsafe_ptr<float>(h*value.stride_at(1)+hs*K+k*8);
                sum += __avx_vec_mul_8_fp32(A, B);
            }
            for (int32_t k = 8*_8_chunk; k < K; ++k) {
                float a = attn_score.data_at<float>(h*attn_score.stride_at(0)+t*attn_score.stride_at(1)+k);
                float b = value.data_at<float>(h*value.stride_at(1)+hs*K+k);
                sum += a*b;
            }
            uint32_t index = t*attn_output.stride_at(1)+h*head_size+hs;
            *attn_output.unsafe_ptr<float>(index) = sum;
        }
    }
}

void _avx_GELU_fp32(SchedParam sched_param, const Tensor& input, Tensor& out) {
    const float* src = input.unsafe_ptr<float>(sched_param.this_thread_begin_index()*8);
    float* dst =  out.unsafe_ptr<float>(sched_param.this_thread_begin_index()*8);
    float parameters[8] = {0.044715f, 0.79788458f, 378.f, 17325.f, 135135.f, 28.f, 3150.f, 62370.f};
    _avx_GELU_fp32(dst, src, 8, parameters);
}

void _avx_norm_fp32(SchedParam sched_param, const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& out, NormParam& norm_param) {
    const int32_t nbs = input.stride_at(1);
    const float* src = input.unsafe_ptr<float>(sched_param.this_thread_begin_index()*nbs);
    const float* gamma = weight.unsafe_ptr<float>(0);
    const float* beta = bias.unsafe_ptr<float>(0);
    float* dst =  out.unsafe_ptr<float>(sched_param.this_thread_begin_index()*nbs);
    _avx_norm_fp32(dst, src, gamma, beta, norm_param.epsilon, input.dim_at(2), false);
}

} // namespace mariana
