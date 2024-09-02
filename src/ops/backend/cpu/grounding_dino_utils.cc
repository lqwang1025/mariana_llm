/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/grounding_dino_utils.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-07-10:10:13:36
 * Description:
 * 
 */

#include <cmath>
#include <cfloat>

#include <utils/mariana_define.h>

#include <core/impl/allocator.h>

#include <mariana_llm/mariana_llm.h>

#include <ops/grounding_dino_encoder_before.h>
#include <ops/backend/cpu/grounding_dino_utils.h>

namespace mariana {

void grounding_dino_encoder_before(SchedParam sched_param, const tensor_list& inputs, const Tensor& level_embed, Tensor& source_flatten, Tensor& lvl_pos_embed_flatten) {
    size_t levels = inputs.size() / 2;
    std::vector<int32_t> intervals;
    intervals.reserve(levels);
    intervals.push_back(inputs[0].stride_at(1));
    for (size_t i = 1; i < levels; ++i) {
        intervals.push_back(intervals[i-1]+inputs[i].stride_at(1));
    }
    
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        uint32_t idx = i;
        uint32_t idx3 = idx % source_flatten.dim_at(2);
        idx /= source_flatten.dim_at(2);
        uint32_t idx2 = idx % source_flatten.dim_at(1);
        idx /= source_flatten.dim_at(1);
        uint32_t idx1 = idx;
        
        for (size_t level = 0; level < levels; ++level) {
            if (idx2 < (uint32_t)intervals[level]) {
                const int32_t offset    = level == 0 ? 0 : intervals[level-1];
                const int32_t input_idx = idx1*inputs[level].stride_at(0)+idx3*inputs[level].stride_at(1)+idx2-offset;
                *source_flatten.unsafe_ptr<float>(i) = inputs[level].data_at<float>(input_idx);
                *lvl_pos_embed_flatten.unsafe_ptr<float>(i) = inputs[levels+level].data_at<float>(input_idx) + level_embed.data_at<float>(level*level_embed.stride_at(0) + idx3);
                break;
            }
        }
    }
}

void generate_encoder_output_proposals(SchedParam sched_param, ExeContext& context, const Tensor& enc_output, Tensor& output_proposals, Tensor& object_query) {
    GroundingDinoEncoderBeforeFunc::SpatialShapes& sp_shape = *static_cast<GroundingDinoEncoderBeforeFunc::SpatialShapes*>(context.runtime_info.anything);
    IAllocator* allocator = get_allocator(enc_output.device());
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        uint32_t total  = 0;
        uint32_t offset = 0;
        for (uint32_t n = 0; n < sp_shape.size; ++n) {
            total += sp_shape.heights[n]*sp_shape.widths[n];
            if (i < total) {
                uint32_t w = (i-offset)%sp_shape.widths[n];
                uint32_t h = (i-offset)/sp_shape.widths[n];
                float value1 = (w+0.5f)/sp_shape.widths[n];
                bool ok = (value1 > 0.01f) && (value1 < 0.99f);
                
                float value2 = (h+0.5f)/sp_shape.heights[n];
                ok = ok && (value2 > 0.01f) && (value2 < 0.99f);
                                
                float value3 = 0.05f*std::pow(2, n);
                ok = ok && (value3 > 0.01f) && (value3 < 0.99f);
                
                if (!ok) {
                    *output_proposals.unsafe_ptr<float>(i*output_proposals.stride_at(1)+0) = FLT_MAX; // x
                    *output_proposals.unsafe_ptr<float>(i*output_proposals.stride_at(1)+1) = FLT_MAX; // y
                    *output_proposals.unsafe_ptr<float>(i*output_proposals.stride_at(1)+2) = FLT_MAX;
                    *output_proposals.unsafe_ptr<float>(i*output_proposals.stride_at(1)+3) = FLT_MAX;
                    float* objq_ptr   = object_query.unsafe_ptr<float>(i*object_query.stride_at(1));
                    memset(objq_ptr, 0, sizeof(float)*object_query.dim_at(2));
                    for (uint32_t l = 0; l < (uint32_t)object_query.dim_at(2); ++l) {
                        *object_query.unsafe_ptr<float>(i*object_query.stride_at(1)+l) = 0.f;
                    }
                } else {
                    value1 = log(value1/(1-value1));
                    value2 = log(value2/(1-value2));
                    value3 = log(value3/(1-value3));
                    *output_proposals.unsafe_ptr<float>(i*output_proposals.stride_at(1)+0) = value1; // x
                    *output_proposals.unsafe_ptr<float>(i*output_proposals.stride_at(1)+1) = value2; // y
                    *output_proposals.unsafe_ptr<float>(i*output_proposals.stride_at(1)+2) = value3;
                    *output_proposals.unsafe_ptr<float>(i*output_proposals.stride_at(1)+3) = value3;
                    float* objq_ptr   = object_query.unsafe_ptr<float>(i*object_query.stride_at(1));
                    float* encout_ptr = enc_output.unsafe_ptr<float>(i*object_query.stride_at(1));
                    allocator->memcpy(objq_ptr, encout_ptr, sizeof(float)*object_query.dim_at(2));
                }
                break;
                
            }
            offset += sp_shape.heights[n]*sp_shape.widths[n];
        }
    }
}

void decoder_reference_points_correct(SchedParam sched_param, const Tensor& tmp, const Tensor& reference_points, Tensor& out, float eps) {
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        float a = tmp.data_at<float>(i);
        float b = reference_points.data_at<float>(i);
        if (b < eps) {
            b = eps;
        } else if (b > 1-eps) {
            b = 1-eps;
        }
        float c = log(b/(1-b))+a;
        *out.unsafe_ptr<float>(i) = 1.f/(1+expf(-c));
    }
}

void bbox_center_to_corners(SchedParam sched_param, const Tensor& src, int32_t img_h, int32_t img_w, Tensor& out) {
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        float cx = src.data_at<float>(i*src.stride_at(1)+0);
        float cy = src.data_at<float>(i*src.stride_at(1)+1);
        float w  = src.data_at<float>(i*src.stride_at(1)+2);
        float h  = src.data_at<float>(i*src.stride_at(1)+3);
        
        *out.unsafe_ptr<float>(i*src.stride_at(1)+0) = (cx-0.5f*w)*img_w;
        *out.unsafe_ptr<float>(i*src.stride_at(1)+1) = (cy-0.5f*h)*img_h;
        *out.unsafe_ptr<float>(i*src.stride_at(1)+2) = (cx+0.5f*w)*img_w;
        *out.unsafe_ptr<float>(i*src.stride_at(1)+3) = (cy+0.5f*h)*img_h;
    }
}

void grounding_dino_pre_process(SchedParam sched_param, const Tensor& src, const std::vector<float>& means, const std::vector<float>& stds, Tensor& out) {
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
    }
}

} // namespace mariana
