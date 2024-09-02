/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/sched_param.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-20:20:26:01
 * Description:
 *
 */

#ifndef __OPS_SCHED_PARAM_H__
#define __OPS_SCHED_PARAM_H__

#include <cstdint>

namespace mariana {

struct SchedParam { 
    uint64_t n_thread = 0;
    uint64_t i_thread = 0;
    uint64_t n_offset = 0;
    uint64_t n_chunk  = 0;

    uint64_t this_thread_begin_index() {
        return n_offset+i_thread*n_chunk;
    }

    uint64_t this_thread_end_index() {
        return this_thread_begin_index()+n_chunk;
    }
};
    
} // namespace mariana

#endif /* __OPS_SCHED_PARAM_H__ */

