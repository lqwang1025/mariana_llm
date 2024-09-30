/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/mhs_attention.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-09-27:10:38:53
 * Description:
 * 
 */

#include <cfloat>

#include <ops/backend/gpu/impl/mhs_attention.h>

namespace mariana {

__global__ void __mhs_mask_attention_float32_kernel(const float* q_ptr, const float* k_ptr, const float* v_ptr, const float* mask_ptr, uint32_t QT, uint32_t KT, uint32_t VT, uint32_t n_head, uint32_t head_size, uint32_t q_stride_1, uint32_t k_stride_1, uint32_t mask_dim_1, uint32_t mask_stride_1, uint32_t mask_stride_2, uint32_t v_stride_1, uint32_t out_stride_1, float* out_ptr) {
    int32_t t = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (t >= QT) return;
    float scale = 1.f/sqrtf(head_size);
    float _qk_dot[256] = {0.f};
    for (uint32_t t = 0; t < QT; ++t) {
        for (int32_t h = 0; h < n_head; ++h) {
            //1. Q@V
            float maxval = -FLT_MAX;
            for (uint32_t t2 = 0; t2 < KT; ++t2) { // 52
                float val = 0.f;
                for (int i = 0; i < head_size; ++i) { // 64
                    float q = q_ptr[t*q_stride_1+h*head_size+i];
                    float k = k_ptr[t2*k_stride_1+h*head_size+i];
                    val += q*k;
                }
                uint32_t mask_id = t*mask_stride_2+t2;
                if (mask_dim_1 == n_head) {
                    mask_id += h*mask_stride_1;
                }
                _qk_dot[t2] = val*scale+mask_ptr[mask_id];
                if (_qk_dot[t2] > maxval) {
                    maxval = _qk_dot[t2];
                }
            }
            
            //2. softmax
            float expsum = 0.f;
            for (uint32_t t2 = 0; t2 < KT; ++t2) { // 52
                float expv = expf(_qk_dot[t2]-maxval);
                expsum += expv;
                _qk_dot[t2] = expv;
            }
            float expsum_inv = expsum == 0.f ? 0.f : 1.f/expsum;
            for (uint32_t t2 = 0; t2 < KT; ++t2) { // 52
                _qk_dot[t2] *= expsum_inv;
            }

            // 3. attention_scores @ V
            for (int32_t i = 0; i < head_size; ++i) {
                out_ptr[t*out_stride_1+h*head_size+i] = 0;
            }
            for (uint32_t t2 = 0; t2 < VT; ++t2) {
                for (int32_t i = 0; i < head_size; ++i) {
                    float value = v_ptr[t2*v_stride_1+h*head_size+i];
                    out_ptr[t*out_stride_1+h*head_size+i] += value*_qk_dot[t2];
                }
            }
        } // n_head
    } // NC
}

__global__ void __mhs_attention_float32_kernel(const float* q_ptr, const float* k_ptr, const float* v_ptr, uint32_t QT, uint32_t KT, uint32_t VT, uint32_t n_head, uint32_t head_size, uint32_t q_stride_1, uint32_t k_stride_1, uint32_t v_stride_1, uint32_t out_stride_1, float* out_ptr) {
    int32_t t = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (t >= QT) return;
    float scale = 1.f/sqrtf(head_size);
    float _qk_dot[256] = {0.f};
    for (uint32_t t = 0; t < QT; ++t) {
        for (int32_t h = 0; h < n_head; ++h) {
            //1. Q@V
            float maxval = -FLT_MAX;
            for (uint32_t t2 = 0; t2 < KT; ++t2) { // 52
                float val = 0.f;
                for (int i = 0; i < head_size; ++i) { // 64
                    float q = q_ptr[t*q_stride_1+h*head_size+i];
                    float k = k_ptr[t2*k_stride_1+h*head_size+i];
                    val += q*k;
                }
                _qk_dot[t2] = val*scale;
                if (_qk_dot[t2] > maxval) {
                    maxval = _qk_dot[t2];
                }
            }
            
            //2. softmax
            float expsum = 0.f;
            for (uint32_t t2 = 0; t2 < KT; ++t2) { // 52
                float expv = expf(_qk_dot[t2]-maxval);
                expsum += expv;
                _qk_dot[t2] = expv;
            }
            float expsum_inv = expsum == 0.f ? 0.f : 1.f/expsum;
            for (uint32_t t2 = 0; t2 < KT; ++t2) { // 52
                _qk_dot[t2] *= expsum_inv;
            }

            // 3. attention_scores @ V
            for (int32_t i = 0; i < head_size; ++i) {
                out_ptr[t*out_stride_1+h*head_size+i] = 0;
            }
            for (uint32_t t2 = 0; t2 < VT; ++t2) {
                for (int32_t i = 0; i < head_size; ++i) {
                    float value = v_ptr[t2*v_stride_1+h*head_size+i];
                    out_ptr[t*out_stride_1+h*head_size+i] += value*_qk_dot[t2];
                }
            }
        } // n_head
    } // NC
}

__global__ void __mhs_swin_mask_attention_float32_kernel(const float* q_ptr, const float* k_ptr, const float* v_ptr, const float* mask_ptr, const float* pos_mask_ptr, uint32_t QT, uint32_t KT, uint32_t VT, uint32_t n_head, uint32_t head_size, uint32_t q_stride_1, uint32_t k_stride_1, uint32_t mask_stride_2, uint32_t pmask_dim_1, uint32_t pmask_stride_1, uint32_t pmask_stride_2, uint32_t v_stride_1, uint32_t out_stride_1, float* out_ptr) {
    int32_t t = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (t >= QT) return;
    float scale = 1.f/sqrtf(head_size);
    float _qk_dot[256] = {0.f};
    for (uint32_t t = 0; t < QT; ++t) {
        for (int32_t h = 0; h < n_head; ++h) {
            //1. Q@V
            float maxval = -FLT_MAX;
            for (uint32_t t2 = 0; t2 < KT; ++t2) { // 52
                float val = 0.f;
                for (int i = 0; i < head_size; ++i) { // 64
                    float q = q_ptr[t*q_stride_1+h*head_size+i];
                    float k = k_ptr[t2*k_stride_1+h*head_size+i];
                    val += q*k;
                }
                uint32_t mask_id = t*pmask_stride_2+t2;
                if (pmask_dim_1 == n_head) {
                    mask_id += h*pmask_stride_1;
                }
                uint32_t att_mask_id = t*mask_stride_2+t2;
                _qk_dot[t2] = val*scale+pos_mask_ptr[mask_id]+mask_ptr[att_mask_id];
                if (_qk_dot[t2] > maxval) {
                    maxval = _qk_dot[t2];
                }
            }
            
            //2. softmax
            float expsum = 0.f;
            for (uint32_t t2 = 0; t2 < KT; ++t2) { // 52
                float expv = expf(_qk_dot[t2]-maxval);
                expsum += expv;
                _qk_dot[t2] = expv;
            }
            float expsum_inv = expsum == 0.f ? 0.f : 1.f/expsum;
            for (uint32_t t2 = 0; t2 < KT; ++t2) { // 52
                _qk_dot[t2] *= expsum_inv;
            }

            // 3. attention_scores @ V
            for (int32_t i = 0; i < head_size; ++i) {
                out_ptr[t*out_stride_1+h*head_size+i] = 0;
            }
            for (uint32_t t2 = 0; t2 < VT; ++t2) {
                for (int32_t i = 0; i < head_size; ++i) {
                    float value = v_ptr[t2*v_stride_1+h*head_size+i];
                    out_ptr[t*out_stride_1+h*head_size+i] += value*_qk_dot[t2];
                }
            }
        } // n_head
    } // NC
}

void mhs_mask_attention(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& mask, Tensor& out, int32_t n_head, int32_t head_size, CUDAContext* cuda_ctx) {
    if (out.dtype().match<float>()) {
        const uint32_t QT = Q.dim_at(1); // [batch, T, C]
        const uint32_t KT = K.dim_at(1); // [batch, T, C]
        const uint32_t VT = V.dim_at(1); // [batch, T, C]
        const uint32_t q_offset = Q.stride_at(0);
        const uint32_t k_offset = K.stride_at(0);
        const uint32_t v_offset = V.stride_at(0);
        const uint32_t m_offset = mask.stride_at(0);
        const uint32_t o_offset = out.stride_at(0);
        float* q_ptr    = Q.unsafe_ptr<float>(sched_param.this_thread_begin_index()*q_offset);
        float* k_ptr    = K.unsafe_ptr<float>(sched_param.this_thread_begin_index()*k_offset);
        float* v_ptr    = V.unsafe_ptr<float>(sched_param.this_thread_begin_index()*v_offset);
        float* mask_ptr = mask.unsafe_ptr<float>(sched_param.this_thread_begin_index()*m_offset);
        float* out_ptr  = out.unsafe_ptr<float>(sched_param.this_thread_begin_index()*o_offset);
        __mhs_mask_attention_float32_kernel<<<get_cuda_gridsize(QT, CUDA_ATTN_BLOCK_SIZE),
            CUDA_ATTN_BLOCK_SIZE, 0, cuda_ctx->stream(sched_param.id_thread)>>>(q_ptr, k_ptr, v_ptr, mask_ptr, QT, KT, VT, n_head, head_size, Q.stride_at(1), K.stride_at(1), mask.dim_at(1), mask.stride_at(1), mask.stride_at(2), V.stride_at(1), out.stride_at(1), out_ptr);
    } else {
        MLOG(FATAL)<<"Mhs attention unsupport datatype:"<<out.dtype().name();
    }
}

void mhs_attention(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& out, int32_t n_head, int32_t head_size, CUDAContext* cuda_ctx) {
    if (out.dtype().match<float>()) {
        const uint32_t QT = Q.dim_at(1); // [batch, T, C]
        const uint32_t KT = K.dim_at(1); // [batch, T, C]
        const uint32_t VT = V.dim_at(1); // [batch, T, C]
        const uint32_t q_offset = Q.stride_at(0);
        const uint32_t k_offset = K.stride_at(0);
        const uint32_t v_offset = V.stride_at(0);
        const uint32_t o_offset = out.stride_at(0);
        float* q_ptr    = Q.unsafe_ptr<float>(sched_param.this_thread_begin_index()*q_offset);
        float* k_ptr    = K.unsafe_ptr<float>(sched_param.this_thread_begin_index()*k_offset);
        float* v_ptr    = V.unsafe_ptr<float>(sched_param.this_thread_begin_index()*v_offset);
        float* out_ptr  = out.unsafe_ptr<float>(sched_param.this_thread_begin_index()*o_offset);
        __mhs_attention_float32_kernel<<<get_cuda_gridsize(QT, CUDA_ATTN_BLOCK_SIZE),
            CUDA_ATTN_BLOCK_SIZE, 0, cuda_ctx->stream(sched_param.id_thread)>>>(q_ptr, k_ptr, v_ptr, QT, KT, VT, n_head, head_size, Q.stride_at(1), K.stride_at(1), V.stride_at(1), out.stride_at(1), out_ptr);
    } else {
        MLOG(FATAL)<<"Mhs attention unsupport datatype:"<<out.dtype().name();
    }
}

void mhs_swin_mask_attention(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& pos_mask, const Tensor& attn_mask, Tensor& out, int32_t n_head, int32_t head_size, CUDAContext* cuda_ctx) {
    if (out.dtype().match<float>()) {
        const uint32_t QT = Q.dim_at(1); // [batch, T, C]
        const uint32_t KT = K.dim_at(1); // [batch, T, C]
        const uint32_t VT = V.dim_at(1); // [batch, T, C]
        const uint32_t q_offset  = Q.stride_at(0);
        const uint32_t k_offset  = K.stride_at(0);
        const uint32_t v_offset  = V.stride_at(0);
        const uint32_t m_offset  = attn_mask.stride_at(0);
        const uint32_t pm_offset = pos_mask.stride_at(0);
        const uint32_t o_offset  = out.stride_at(0);
        float* q_ptr     = Q.unsafe_ptr<float>(sched_param.this_thread_begin_index()*q_offset);
        float* k_ptr     = K.unsafe_ptr<float>(sched_param.this_thread_begin_index()*k_offset);
        float* v_ptr     = V.unsafe_ptr<float>(sched_param.this_thread_begin_index()*v_offset);
        float* mask_ptr  = attn_mask.unsafe_ptr<float>(sched_param.this_thread_begin_index()*m_offset);
        float* pmask_ptr = pos_mask.unsafe_ptr<float>(sched_param.this_thread_begin_index()*pm_offset);
        float* out_ptr   = out.unsafe_ptr<float>(sched_param.this_thread_begin_index()*o_offset);
        __mhs_swin_mask_attention_float32_kernel<<<get_cuda_gridsize(QT, CUDA_ATTN_BLOCK_SIZE),
            CUDA_ATTN_BLOCK_SIZE, 0, cuda_ctx->stream(sched_param.id_thread)>>>(q_ptr, k_ptr, v_ptr, mask_ptr, pmask_ptr, QT, KT, VT, n_head, head_size, Q.stride_at(1), K.stride_at(1), attn_mask.stride_at(2), pos_mask.dim_at(1), pos_mask.stride_at(1), pos_mask.stride_at(2), V.stride_at(1), out.stride_at(1), out_ptr);
    } else {
        MLOG(FATAL)<<"MHS SWIN attention unsupport datatype:"<<out.dtype().name();
    }
}

} // namespace mariana
