/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/pass.cc
 * Authors    : lqwang@pandora
 * Create Time: 2024-07-11:06:01:55
 * Description:
 * 
 */

#include <ops/pass.h>

namespace mariana {

bool PassFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    // warning on here
    Tensor tensor(inputs[0].dims(), inputs[0].device(), inputs[0].unsafe_ptr<uint8_t>(0), inputs[0].dtype());
    if (outputs.empty()) {
        outputs.push_back(tensor);
    } else {
        outputs[0] = tensor;
    }
    return true;
}

bool PassFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    // do nothing
    return true;
}

} // namespace mariana
