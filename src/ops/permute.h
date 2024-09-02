/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/permute.h
 * Authors    : lqwang@inspur
 * Create Time: 2024-06-24:09:24:42
 * Description:
 *
 */

#ifndef __OPS_PERMUTE_H__
#define __OPS_PERMUTE_H__

#include <core/function.h>

namespace mariana {

struct PermuteFunc : public Function {
    bool init(const ModelParam& param, const std::string& node_name)override;
    bool plan_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
protected:
    bool _forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context)override;
private:
    uint8_t m_perms[4] = {0, 0, 0, 0};
};

} // namespace mariana

#endif /* __OPS_PERMUTE_H__ */

