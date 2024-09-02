/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/get_rows.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-18:11:54:27
 * Description:
 *
 */

#ifndef __OPS_GET_ROWS_H__
#define __OPS_GET_ROWS_H__

#include <core/tensor.h>
#include <core/function.h>

namespace mariana {

struct GetRowsFunc : public Function {
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
private:
    Tensor m_weight;
};

} // namespace mariana

#endif /* __OPS_GET_ROWS_H__ */

