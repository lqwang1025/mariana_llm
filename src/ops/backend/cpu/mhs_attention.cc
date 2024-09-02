/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/mhs_attention.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-25:06:28:57
 * Description:
 * 
 */

#include <cmath>
#include <cfloat>

#include <ops/backend/cpu/mhs_attention.h>
#include <ops/backend/cpu/x86_x64/avx/avx_funcs.h>

namespace mariana {

[[maybe_unused]]static void _navie_mhs_mask_attention_batch_split(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& mask, Tensor& out, int32_t n_head, int32_t head_size) {
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
                for (int i = 0; i < head_size; ++i) { // 64
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
            
            //2. softmax
            float expsum = 0.f;
            for (uint32_t t2 = 0; t2 < T; ++t2) { // 52
                float expv = expf(_qk_dot[t2]-maxval);
                expsum += expv;
                _qk_dot[t2] = expv;
            }
            float expsum_inv = expsum == 0.f ? 0.f : 1.f/expsum;
            for (uint32_t t2 = 0; t2 < T; ++t2) { // 52
                _qk_dot[t2] *= expsum_inv;
            }

            // 3. attention_scores @ V
            for (int32_t i = 0; i < head_size; ++i) {
                *out.unsafe_ptr<float>(b*out.stride_at(0)+t*out.stride_at(1)+h*head_size+i) = 0;
            }
            for (uint32_t t2 = 0; t2 < T; ++t2) {
                for (int32_t i = 0; i < head_size; ++i) {
                    float value = V.data_at<float>(b*V.stride_at(0)+t2*V.stride_at(1)+h*head_size+i);
                    *out.unsafe_ptr<float>(b*out.stride_at(0)+t*out.stride_at(1)+h*head_size+i) += value*_qk_dot[t2];
                }
            }
        } // n_head
    }
}

[[maybe_unused]]static void _navie_mhs_attention_batch_split(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& out, int32_t n_head, int32_t head_size) {
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
                for (int i = 0; i < head_size; ++i) { // 64
                    float q = Q.data_at<float>(b*Q.stride_at(0)+t*Q.stride_at(1)+h*head_size+i);
                    float k = K.data_at<float>(b*K.stride_at(0)+t2*K.stride_at(1)+h*head_size+i);
                    val += q*k;
                }
                _qk_dot[t2] = val*scale;
                if (_qk_dot[t2] > maxval) {
                    maxval = _qk_dot[t2];
                }
            }
            
            //2. softmax
            float expsum = 0.f;
            for (uint32_t t2 = 0; t2 < T; ++t2) { // 52
                float expv = expf(_qk_dot[t2]-maxval);
                expsum += expv;
                _qk_dot[t2] = expv;
            }
            float expsum_inv = expsum == 0.f ? 0.f : 1.f/expsum;
            for (uint32_t t2 = 0; t2 < T; ++t2) { // 52
                _qk_dot[t2] *= expsum_inv;
            }

            // 3. attention_scores @ V
            for (int32_t i = 0; i < head_size; ++i) {
                *out.unsafe_ptr<float>(b*out.stride_at(0)+t*out.stride_at(1)+h*head_size+i) = 0;
            }
            for (uint32_t t2 = 0; t2 < VT; ++t2) {
                for (int32_t i = 0; i < head_size; ++i) {
                    float value = V.data_at<float>(b*V.stride_at(0)+t2*V.stride_at(1)+h*head_size+i);
                    *out.unsafe_ptr<float>(b*out.stride_at(0)+t*out.stride_at(1)+h*head_size+i) += value*_qk_dot[t2];
                }
            }
        } // n_head
    }
}

static void _navie_mhs_mask_attention_head_split(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& mask, Tensor& out, int32_t n_head, int32_t head_size) {
    float scale = 1.f/sqrtf(head_size);
    uint32_t B = Q.dim_at(0);
    uint32_t T = Q.dim_at(1); // 52
    float _qk_dot[T] = {0.f};
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t h = sched_param.this_thread_begin_index(); h < sched_param.this_thread_end_index(); ++h) { // 12

                //1. Q@V
                float maxval = -FLT_MAX;
                for (uint32_t t2 = 0; t2 < T; ++t2) { // 52
                    float val = 0.f;
                    for (int i = 0; i < head_size; ++i) { // 64
                        float q = Q.data_at<float>(b*Q.stride_at(0)+t*Q.stride_at(1)+h*head_size+i);
                        float k = K.data_at<float>(b*K.stride_at(0)+t2*K.stride_at(1)+h*head_size+i);
                        val += q*k;
                    }
                    uint32_t mask_id = t*mask.stride_at(2)+t2;
                    if ((uint32_t)mask.dim_at(0) == B) {
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
                //2. softmax
                float expsum = 0.f;
                for (uint32_t t2 = 0; t2 < T; ++t2) { // 52
                    float expv = expf(_qk_dot[t2]-maxval);
                    expsum += expv;
                    _qk_dot[t2] = expv;
                }
                float expsum_inv = expsum == 0.f ? 0.f : 1.f/expsum;
                for (uint32_t t2 = 0; t2 < T; ++t2) { // 52
                    _qk_dot[t2] *= expsum_inv;
                }

                // 3. attention_scores @ V
                for (int32_t i = 0; i < head_size; ++i) {
                    *out.unsafe_ptr<float>(b*out.stride_at(0)+t*out.stride_at(1)+h*head_size+i) = 0;
                }
                for (uint32_t t2 = 0; t2 < T; ++t2) {
                    for (int32_t i = 0; i < head_size; ++i) {
                        float value = V.data_at<float>(b*V.stride_at(0)+t2*V.stride_at(1)+h*head_size+i);
                        *out.unsafe_ptr<float>(b*out.stride_at(0)+t*out.stride_at(1)+h*head_size+i) += value*_qk_dot[t2];
                    }
                }
            }
        }
    }
}

[[maybe_unused]]static void _navie_mhs_swin_mask_attention_batch_split(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& pos_mask, const Tensor& attn_mask, Tensor& out, int32_t n_head, int32_t head_size) {
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
                for (int i = 0; i < head_size; ++i) { // 64
                    float q = Q.data_at<float>(b*Q.stride_at(0)+t*Q.stride_at(1)+h*head_size+i);
                    float k = K.data_at<float>(b*K.stride_at(0)+t2*K.stride_at(1)+h*head_size+i);
                    val += q*k;
                }
                uint32_t mask_id = t*pos_mask.stride_at(2)+t2;
                if (pos_mask.dim_at(0) == B) {
                    mask_id += b*pos_mask.stride_at(0);
                }
                if (pos_mask.dim_at(1) == n_head) {
                    mask_id += h*pos_mask.stride_at(1);
                }
                uint32_t att_mask_id = b*attn_mask.stride_at(0)+t*attn_mask.stride_at(2)+t2;

                _qk_dot[t2] = val*scale+pos_mask.data_at<float>(mask_id)+attn_mask.data_at<float>(att_mask_id);
                if (_qk_dot[t2] > maxval) {
                    maxval = _qk_dot[t2];
                }
            }
            
            //2. softmax
            float expsum = 0.f;
            for (uint32_t t2 = 0; t2 < T; ++t2) { // 52
                float expv = expf(_qk_dot[t2]-maxval);
                expsum += expv;
                _qk_dot[t2] = expv;
            }
            float expsum_inv = expsum == 0.f ? 0.f : 1.f/expsum;
            for (uint32_t t2 = 0; t2 < T; ++t2) { // 52
                _qk_dot[t2] *= expsum_inv;
            }

            // 3. attention_scores @ V
            for (int32_t i = 0; i < head_size; ++i) {
                *out.unsafe_ptr<float>(b*out.stride_at(0)+t*out.stride_at(1)+h*head_size+i) = 0;
            }
            for (uint32_t t2 = 0; t2 < T; ++t2) {
                for (int32_t i = 0; i < head_size; ++i) {
                    float value = V.data_at<float>(b*V.stride_at(0)+t2*V.stride_at(1)+h*head_size+i);
                    *out.unsafe_ptr<float>(b*out.stride_at(0)+t*out.stride_at(1)+h*head_size+i) += value*_qk_dot[t2];
                }
            }
        } // n_head
    }
}

[[maybe_unused]] static void _navie_bimhs_qk_dot_batch_split(SchedParam sched_param, const Tensor& Q, const Tensor& K, Tensor& attn_weights, int32_t n_head, int32_t head_size, std::atomic<float>& max_val) {
    uint32_t T = Q.dim_at(1);
    uint32_t key_t = K.dim_at(1); // 52
    for (uint32_t row = sched_param.this_thread_begin_index(); row < sched_param.this_thread_end_index(); ++row) {
        uint32_t b = row/T;
        uint32_t t = row%T; // 20906
        for (int32_t h = 0; h < n_head; ++h) { // 4
            //1. Q@V
            for (uint32_t t2 = 0; t2 < key_t; ++t2) { // 52
                float val = 0.f;
                for (int i = 0; i < head_size; ++i) { // 256
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

[[maybe_unused]] static void _naive_bimhs_grdino_attn_clamp_bt_split(SchedParam sched_param, Tensor& attn_weights, Tensor& text_attn_weights, const float& max_val, float clamp_min_val, float clamp_max_val) {
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
            float expsum = 0.f;
            for (uint32_t t = 0; t < T; ++t) { // 20906
                uint32_t index = b*text_attn_weights.stride_at(0)+c*text_attn_weights.stride_at(1)+t;
                float attn_weight = *text_attn_weights.unsafe_ptr<float>(index);
                float expv = expf(attn_weight-row_max_vals[c]);
                expsum += expv;
                *text_attn_weights.unsafe_ptr<float>(index) = expv;
            }
            float expsum_inv = expsum == 0.f ? 0.f : 1.f/expsum;
            for (uint32_t t = 0; t < T; ++t) { // 20906
                uint32_t index = b*text_attn_weights.stride_at(0)+c*text_attn_weights.stride_at(1)+t;
                *text_attn_weights.unsafe_ptr<float>(index) *= expsum_inv;
            }
        }

        // vision_attn_weight softmax
        for (uint32_t t = 0; t < T; ++t) { // 20906
            float expsum = 0.f;
            for (uint32_t c = 0; c < C; ++c) { // 52
                uint32_t index = b*attn_weights.stride_at(0)+t*attn_weights.stride_at(1)+c;
                float attn_weight = *attn_weights.unsafe_ptr<float>(index);
                float expv = expf(attn_weight-max_val);
                expsum += expv;
                *attn_weights.unsafe_ptr<float>(index) = expv;
            }
            float expsum_inv = expsum == 0.f ? 0.f : 1.f/expsum;
            for (uint32_t c = 0; c < C; ++c) { // 52
                uint32_t index = b*attn_weights.stride_at(0)+t*attn_weights.stride_at(1)+c;
                *attn_weights.unsafe_ptr<float>(index) *= expsum_inv;
            }
        }
    }
}

[[maybe_unused]] static void _naive_bimhs_grdino_attn_v_dot_batch_split(SchedParam sched_param, const Tensor& attn_score, const Tensor& value, Tensor& attn_output, int32_t n_head, int32_t head_size) {
    // input1: 4 20906 52 input2: [1, 4 256 52] out: 4 20906 256
    int32_t T  = attn_score.dim_at(1); // 20906
    int32_t K  = attn_score.dim_at(2); // 52
        
    for (uint32_t row = sched_param.this_thread_begin_index(); row < sched_param.this_thread_end_index(); ++row) {
        uint32_t h = row/T;
        uint32_t t = row%T;
        for (uint32_t hs = 0; hs < (uint32_t)head_size; ++hs) { // 256
            float sum = 0.f;
            for (uint32_t k = 0; k < (uint32_t)K; ++k) { // 52
                float attn_prob = attn_score.data_at<float>(h*attn_score.stride_at(0)+t*attn_score.stride_at(1)+k);
                float v         = value.data_at<float>(h*value.stride_at(1)+hs*K+k);
                sum += (attn_prob*v);
            }
            uint32_t index = t*attn_output.stride_at(1)+h*head_size+hs;
            *attn_output.unsafe_ptr<float>(index) = sum;
        }
    }
}

void _naive_multi_scale_deformable_attention(SchedParam sched_param, const GroundingDinoEncoderBeforeFunc::SpatialShapes& spatial_shapes, const Tensor& value, const Tensor& attention_weights, const Tensor& sampling_offset, const Tensor& reference_points, Tensor& out) {
    
    int32_t C  = out.dim_at(1);
    
    // out dims: 1 20906 256=num_heads*hidden_dim
    // value : shape: [1, 20906, 8, 32] view as [8, 32, sw, sh]
    // sampling_offset: shape: [1, 20906, 8, 4, 4, 2] view as [8, 20906, 4, 2], split by 3rd dim
    // reference_points: shape: [1, 20906, 1, 4, 1, 2]
    // attention_weights: shape: [1, 20906, 8, 4, 4]
    int32_t num_heads   = value.dim_at(2);
    int32_t hidden_dim  = value.dim_at(3);
    int32_t num_levels  = sampling_offset.dim_at(3);
    int32_t num_points  = sampling_offset.dim_at(4);
    // For each output location output[n, :, h, w], the size-2 vector grid[n, h, w]
    // specifies input pixel locations x and y, which are used to interpolate the output value output[n, :, h, w].
    // In the case of 5D inputs, grid[n, d, h, w] specifies the x, y, z pixel locations for interpolating output[n, :, d, h, w].
    // mode argument specifies nearest or bilinear interpolation method to sample the input pixels.
    for (uint32_t h = 0; h < (uint32_t)num_heads; ++h) { // 8
        for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) { // grid_sample output shape is : [8, 32, 20906, 4x4]
            uint32_t b = i/C;
            uint32_t c = i%C;           
            for (uint32_t hd = 0; hd < (uint32_t)hidden_dim; ++hd) { // 32
                float sum = 0.f;
                uint32_t height = 0;
                uint32_t width  = 0;
                uint32_t offset = 0;
                for (uint32_t nl = 0; nl < (uint32_t)num_levels; ++nl) { // 4
                    height = spatial_shapes.heights[nl];
                    width  = spatial_shapes.widths[nl];
                    for (uint32_t np = 0; np < (uint32_t)num_points; ++np) { // 4
                        uint32_t so_index = b*sampling_offset.stride_at(0)+c*sampling_offset.stride_at(1)
                            +h*sampling_offset.stride_at(2)+nl*sampling_offset.stride_at(3)+np*sampling_offset.stride_at(4);
                        float x = 0.f, y = 0.f;
                        if (reference_points.dim_at(5) == 2) {
                            uint32_t ref_index = b*reference_points.stride_at(0)+c*reference_points.stride_at(1)+nl*reference_points.stride_at(3);
                            float ref_point = reference_points.data_at<float>(ref_index);
                            x = 2*(sampling_offset.data_at<float>(so_index+0)/width+ref_point)-1;
                            ref_point = reference_points.data_at<float>(ref_index+1);
                            y = 2*(sampling_offset.data_at<float>(so_index+1)/height+ref_point)-1;
                        } else if (reference_points.dim_at(5) == 4) {
                            uint32_t ref_index = b*reference_points.stride_at(0)+c*reference_points.stride_at(1);
                            float ref_point = reference_points.data_at<float>(ref_index+2);
                            x = sampling_offset.data_at<float>(so_index+0)/num_points*ref_point*0.5f;
                            ref_point = reference_points.data_at<float>(ref_index+3);
                            y = sampling_offset.data_at<float>(so_index+1)/num_points*ref_point*0.5f;
                            ref_point = reference_points.data_at<float>(ref_index+0);
                            x += ref_point;
                            x = 2*x -1;
                            ref_point = reference_points.data_at<float>(ref_index+1);
                            y += ref_point;
                            y = 2*y -1;
                        }
                        
                        // when align_corners = true  x = (x+1)*(width-1)/2;  y = (y+1)*(height-1)/2;
                        x = ((x+1)*width-1)/2;
                        y = ((y+1)*height-1)/2;
                        int32_t x1  = std::floor(x+1);
                        int32_t y1  = std::floor(y+1);
                        int32_t x0  = x1 - 1;
                        int32_t y0  = y1 - 1;
                        float   tlv = 0.f;
                        float   blv = 0.f;
                        float   trv = 0.f;
                        float   brv = 0.f;
                        x = std::abs(x-x0);
                        y = std::abs(y-y0);
                        if (0 <= x0 && x0 < (int32_t)width && 0 <= y0 && y0 < (int32_t)height) {
                            uint32_t value_index = b*value.stride_at(0)+(offset+y0*width+x0)*value.stride_at(1)+h*value.stride_at(2)+hd*value.stride_at(3);
                            tlv = value.data_at<float>(value_index);
                            tlv = tlv * (1.f-x)*(1.f-y);
                        }
                        if (0 <= x1 && x1 < (int32_t)width && 0 <= y0 && y0 < (int32_t)height) {
                            uint32_t value_index = b*value.stride_at(0)+(offset+y0*width+x1)*value.stride_at(1)+h*value.stride_at(2)+hd*value.stride_at(3);
                            trv = value.data_at<float>(value_index);
                            trv = trv * x*(1.f-y);
                        
                        }
                        if (0 <= x0 && x0 < (int32_t)width && 0 <= y1 && y1 < (int32_t)height) {
                            uint32_t value_index = b*value.stride_at(0)+(offset+y1*width+x0)*value.stride_at(1)+h*value.stride_at(2)+hd*value.stride_at(3);
                            blv = value.data_at<float>(value_index);
                            blv = blv * (1.f-x)*y;
                        
                        }
                        if (0 <= x1 && x1 < (int32_t)width && 0 <= y1 && y1 < (int32_t)height) {
                            uint32_t value_index = b*value.stride_at(0)+(offset+y1*width+x1)*value.stride_at(1)+h*value.stride_at(2)+hd*value.stride_at(3);
                            brv = value.data_at<float>(value_index);
                            brv = brv * x*y;
                        }
                        uint32_t att_idx = b*attention_weights.stride_at(0)+c*attention_weights.stride_at(1)+h*attention_weights.stride_at(2)+nl*attention_weights.stride_at(3)+np*attention_weights.stride_at(4);
                        float att_weight = attention_weights.data_at<float>(att_idx);
                        sum += (tlv+blv+trv+brv)*att_weight;
                    } // np
                    offset += height*width;
                } // nl
                uint32_t out_idx = b*out.stride_at(0)+c*out.stride_at(1)+h*hidden_dim+hd;
                *out.unsafe_ptr<float>(out_idx) = sum;
            } // hidden_dim
        } //
    } // num_head
}

void bimhs_grdino_qk_dot_batch_split(SchedParam sched_param, const Tensor& Q, const Tensor& K, Tensor& attn_weights, int32_t n_head, int32_t head_size, std::atomic<float>& max_val) {
    //_navie_bimhs_qk_dot_batch_split(sched_param, Q, K, attn_weights, n_head, head_size, max_val);
    _avx2_bimhs_grdino_qk_dot_batch_split(sched_param, Q, K, attn_weights, n_head, head_size, max_val);
}

void bimhs_grdino_attn_v_dot_batch_split(SchedParam sched_param, const Tensor& attn_score, const Tensor& value, Tensor& attn_output, int32_t n_head, int32_t head_size) {
    //_naive_bimhs_grdino_attn_v_dot_batch_split(sched_param, attn_score, value, attn_output, n_head, head_size);
    _avx2_bimhs_grdino_attn_v_dot_batch_split(sched_param, attn_score, value, attn_output, n_head, head_size);
}

void bimhs_grdino_attn_clamp_bt_split(SchedParam sched_param, Tensor& attn_weights, Tensor& text_attn_weights, const float& max_val, float clamp_min_val, float clamp_max_val) {
    //_naive_bimhs_grdino_attn_clamp_bt_split(sched_param, attn_weights, text_attn_weights, max_val, clamp_min_val, clamp_max_val);
    _avx2_bimhs_grdino_attn_clamp_bt_split(sched_param, attn_weights, text_attn_weights, max_val, clamp_min_val, clamp_max_val);
}

void mhs_swin_mask_attention_batch_split(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& pos_mask, const Tensor& attn_mask, Tensor& out, int32_t n_head, int32_t head_size) {
    //_navie_mhs_swin_mask_attention_batch_split(sched_param, Q, K, V, pos_mask, attn_mask, out, n_head, head_size);
    _avx2_mhs_swin_mask_attention_batch_split_fp32(sched_param, Q, K, V, pos_mask, attn_mask, out, n_head, head_size);
}

void mhs_mask_attention_batch_split(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& mask, Tensor& out, int32_t n_head, int32_t head_size) {
    // _navie_mhs_mask_attention_batch_split(sched_param, Q, K, V, mask, out, n_head, head_size);
    _avx2_mhs_mask_attention_batch_split_fp32(sched_param, Q, K, V, mask, out, n_head, head_size);
}

void mhs_attention_batch_split(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, Tensor& out, int32_t n_head, int32_t head_size) {
    // _navie_mhs_attention_batch_split(sched_param, Q, K, V, out, n_head, head_size);
    _avx2_mhs_attention_batch_split_fp32(sched_param, Q, K, V, out, n_head, head_size);
}

void mhs_mask_attention_head_split(SchedParam sched_param, const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& mask, Tensor& out, int32_t n_head, int32_t head_size) {
    _navie_mhs_mask_attention_head_split(sched_param, Q, K, V, mask, out, n_head, head_size);
}

void multi_scale_deformable_attention(SchedParam sched_param, const GroundingDinoEncoderBeforeFunc::SpatialShapes& spatial_shapes, const Tensor& value, const Tensor& attention_weights, const Tensor& sampling_offset, const Tensor& reference_points, Tensor& out) {
    _naive_multi_scale_deformable_attention(sched_param, spatial_shapes, value, attention_weights, sampling_offset, reference_points, out);
}

} // namespace mariana
