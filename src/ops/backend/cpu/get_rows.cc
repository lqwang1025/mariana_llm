/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : get_rows.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-20:08:25:19
 * Description:
 * 
 */

#include <core/impl/allocator.h>

#include <utils/mariana_define.h>
#include <ops/backend/cpu/get_rows.h>

namespace mariana {

void get_rows(SchedParam sched_param, const Tensor& indeices, const Tensor& embedding, Tensor& out) {
    TRACE();
    // const int32_t nth = sched_param.n_thread;
    // const int32_t nb  = indeices.dim_at(0);
    const int32_t nr   = indeices.dim_at(1); // token_size
    const int32_t ne   = embedding.dim_at(1);
    //  const int32_t rows = embedding.dim_at(0);
    //  tensor out shape is : nb, nr ne
    //  tensor embedding shape is : nr ne
    IAllocator* allocator = get_allocator(out.device());
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        int32_t inb = i/nr;
        int32_t inr = i%nr;
        int64_t row_index = 0;
        if (indeices.dtype().match<int32_t>()) {
            row_index = static_cast<int64_t>(indeices.data_at<int32_t>(i));
        } else if (indeices.dtype().match<int64_t>()) {
            row_index = indeices.data_at<int64_t>(i);
        } else {
            MLOG(ERROR)<<"Unsupport dtype:"<<indeices.dtype().name();
            return;
        }
        float* row_ptr = embedding.unsafe_ptr<float>(row_index*ne);
        float* o_row_ptr = out.unsafe_ptr<float>(inb*nr*ne+inr*ne);
        allocator->memcpy(o_row_ptr, row_ptr, sizeof(float)*ne);
    }
}
    
} // namespace mariana
