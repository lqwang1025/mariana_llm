/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/grounding_dino_sine_position_embedding.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-08:11:06:07
 * Description:
 * 
 */

#include <cmath>

#include <utils/mariana_define.h>
#include <ops/backend/cpu/grounding_dino_sine_position_embedding.h>

namespace mariana {

void grounding_dino_sine_position_embedding(SchedParam sched_param, const Tensor& input, Tensor& out, float scale, float temperature) { // input dim is : [n, c, h, w]
    uint32_t embedding_dim = (uint32_t)out.dim_at(1)/2;
    uint32_t H = (uint32_t)out.dim_at(2);
    uint32_t W = (uint32_t)out.dim_at(3);
    const float eps = 1e-6;
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        uint32_t idx = i;
        uint32_t w = idx % out.dim_at(3);
        idx /= out.dim_at(3);
        uint32_t h = idx % out.dim_at(2);
        idx /= out.dim_at(2);
        uint32_t c = idx % out.dim_at(1);
        idx /= out.dim_at(1);
        if (c < embedding_dim) { // pos_y
            float dim_t = c;
            float factor = std::pow(temperature, 2*std::floor(dim_t/2)/embedding_dim);
            float val = (h+1)/(H+eps) * scale;
            // (h+1)/(H+eps) * scale;
            if (c%2 == 0) {
                *out.unsafe_ptr<float>(i) = std::sin(val / factor);
            } else {
                *out.unsafe_ptr<float>(i) = std::cos(val / factor);
            }
        } else { // pos_x
            float dim_t  = c-embedding_dim;
            float factor = std::pow(temperature, 2*std::floor(dim_t/2)/embedding_dim);
            float val = (w+1)/(W+eps) * scale;
            // (h+1)/(H+eps) * scale;
            if (c%2 == 0) {
                *out.unsafe_ptr<float>(i) = std::sin(val / factor);
            } else {
                *out.unsafe_ptr<float>(i) = std::cos(val / factor);
            }
        }
    }
}

void grounding_dino_get_text_enhancer_sine_pos_embed(SchedParam sched_param, const Tensor& input, Tensor& out, float scale, float temperature, bool exchange_xy) {
    // outdim:[1, 52 256] indim:[1, 52, 1]
    // outdim:[1, 900 512] indim:[1, 900, 4]
    int32_t single_level = out.dim_at(2) / input.dim_at(2);

    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        uint32_t idx = i;
        uint32_t c = idx % out.dim_at(2);
        idx /= out.dim_at(2);
        uint32_t t = idx % out.dim_at(1);
        
        int32_t res = c%single_level;
        float factor = std::pow(temperature, 2*std::floor((float)res/2.f)/(float)single_level);
        int32_t num = c/single_level;
        float x = 0.f;
        if (exchange_xy) {
            if (num < 2) {
                num = num == 0 ? 1 : 0;
            }
        }
        if (input.dtype().match<int32_t>()) {
            x = input.data_at<int32_t>(t*input.stride_at(1)+num)*scale/factor;
        } else if (input.dtype().match<float>()) {
            x = input.data_at<float>(t*input.stride_at(1)+num)*scale/factor;
        } else {
            MLOG(FATAL)<<"Unsupport data type in sineposembed:"<<input.dtype().name();
        }

        if (res%2 == 0) {
            *out.unsafe_ptr<float>(i) = std::sin(x);
        } else {
            *out.unsafe_ptr<float>(i) = std::cos(x);
        }
    }
}

} // namespace mariana

