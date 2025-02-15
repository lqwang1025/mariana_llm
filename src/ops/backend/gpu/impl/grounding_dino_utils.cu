/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/grounding_dino_utils.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-15:15:35:25
 * Description:
 * 
 */

#include <ops/backend/gpu/impl/grounding_dino_utils.h>

namespace mariana {

template<typename T>
__global__ void __grounding_dino_encoder_before_4kernel(const T* input, T* out, int32_t odim0, int32_t odim1, int32_t odim2, int32_t odim3, uint32_t input_stride_0, uint32_t input_stride_1, uint32_t input_stride_2, uint32_t input_stride_3, int32_t axis1, int32_t axis2, int32_t step1, int32_t step2, int32_t start1, int32_t start2) {
    
}


void grounding_dino_encoder_before(SchedParam sched_param, const tensor_list& inputs, const Tensor& level_embed, Tensor& source_flatten, Tensor& lvl_pos_embed_flatten, CUDAContext* cuda_ctx) {
    if (source_flatten.dtype().match<float>()) {
        int32_t levels = inputs.size() / 2; // 4
        if (levels == 4) {
            MLOG(INFO)<<"DDDDDDDDDDDDDDDD";
        }
    } else {
        MLOG(FATAL)<<"grounding_dino_encoder_before unsupport datatype:"<<source_flatten.dtype().name();
    }
}

} // namespace mariana
