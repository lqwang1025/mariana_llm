/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/add.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-21:19:36:33
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_ADD_H__
#define __OPS_BACKEND_CPU_ADD_H__

#include <core/tensor.h>
#include <ops/sched_param.h>

namespace mariana {

void add_ele(SchedParam sched_param, const Tensor& a, const Tensor& b, Tensor& out);
    
} // namespace mariana

#endif /* __OPS_BACKEND_CPU_ADD_H__ */

