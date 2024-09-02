/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/swin_patch_merging.cc
 * Authors    : lqwang@pandora
 * Create Time: 2024-07-05:06:45:50
 * Description:
 * 
 */

#include <core/impl/allocator.h>
#include <utils/mariana_define.h>
#include <ops/backend/cpu/swin_patch_merging.h>

namespace mariana {

void swin_patch_merge(SchedParam sched_param, const Tensor& input, Tensor& out, int32_t step) {
    const uint32_t IC = input.dim_at(3);
    IAllocator* allocator = get_allocator(out.device());
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        uint32_t idx = i;
        uint32_t idx4 = idx % 4; // c
        idx /= 4;
        uint32_t idx3 = idx % out.dim_at(2); // w
        idx /= out.dim_at(2);
        uint32_t idx2 = idx % out.dim_at(1); // h
        idx /= out.dim_at(1);
        uint32_t idx1 = idx; // n
        
        void* dst_data = out.unsafe_ptr<float>(i*IC);
        void* src_data = nullptr;
        allocator->memset(dst_data, 0, out.dtype().itemsize()*IC);
        switch (idx4) {
        case 0:
            if (idx2*step < (uint32_t)input.dim_at(1) && idx3*step < (uint32_t)input.dim_at(2)) {
                uint32_t input_offset = idx1*input.stride_at(0) + idx2*step*input.stride_at(1) +
                    idx3*step*input.stride_at(2);
                src_data = input.unsafe_ptr<float>(input_offset);
                allocator->memcpy(dst_data, src_data, input.dtype().itemsize()*IC);
            }
            break;
        case 1:
            if (idx2*step+1 < (uint32_t)input.dim_at(1) && idx3*step < (uint32_t)input.dim_at(2)) {
                uint32_t input_offset = idx1*input.stride_at(0) + (idx2*step+1)*input.stride_at(1) +
                     idx3*step*input.stride_at(2);
                src_data = input.unsafe_ptr<float>(input_offset);
                allocator->memcpy(dst_data, src_data, input.dtype().itemsize()*IC);
            }
            break;
        case 2:
            if (idx2*step < (uint32_t)input.dim_at(1) && idx3*step+1 < (uint32_t)input.dim_at(2)) {
                uint32_t input_offset = idx1*input.stride_at(0) + idx2*step*input.stride_at(1) +
                    (idx3*step+1)*input.stride_at(2);
                src_data = input.unsafe_ptr<float>(input_offset);
                allocator->memcpy(dst_data, src_data, input.dtype().itemsize()*IC);
            }
            break;
        case 3:
            if ((idx2*step+1) < (uint32_t)input.dim_at(1) && (idx3*step+1) < (uint32_t)input.dim_at(2)) {
                uint32_t input_offset = idx1*input.stride_at(0) + (idx2*step+1)*input.stride_at(1) +
                    (idx3*step+1)*input.stride_at(2);
                src_data = input.unsafe_ptr<float>(input_offset);
                allocator->memcpy(dst_data, src_data, input.dtype().itemsize()*IC);
            }
            break;
        default:
            MLOG(ERROR)<<"wrong number of swin patch merge to concat:"<<idx4;
            return;
        }
        
    }
}

} // namespace mariana
