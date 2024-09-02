/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/cpu/get_rows.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-20:08:15:54
 * Description:
 *
 */

#ifndef __OPS_BACKEND_CPU_GET_ROWS_H__
#define __OPS_BACKEND_CPU_GET_ROWS_H__

#include <core/tensor.h>
#include <ops/sched_param.h>

namespace mariana {

void get_rows(SchedParam sched_param, const Tensor& indeices, const Tensor& embedding, Tensor& out);
    
} // namespace mariana

#endif /* __OPS_BACKEND_CPU_GET_ROWS_H__ */

