/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : conv2d.cc
 * Authors    : lqwang@inspur
 * Create Time: 2024-05-28:16:25:12
 * Description:
 * 
 */

#include <ops/conv2d.h>
#include <ops/backend/cpu/im2col.h>
#include <ops/backend/cpu/matmul.h>
#include <ops/backend/cpu/conv2d.h>

#include <models/model_param.h>
#include <utils/mariana_define.h>

namespace mariana {

bool Conv2dFunc::init(const ModelParam& param, const std::string& node_name) {
    ModelParam::SafeTensorInfo sti;
    // TODO: check weight match the parameter.
    TRY_STL(sti = param.sti_map.at(node_name+".weight"), return false);
    Tensor weight(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    TRY_STL(sti = param.sti_map.at(node_name+".bias"), return false);
    Tensor bias(sti.shape, DataOn::CPU, sti.data, sti.dtype);
    m_weight            = weight.deepcopy();
    m_bias              = bias.deepcopy();
    m_act_cate          = param.act_cate;
    m_output_trans      = param.conv_output_trans;
    m_param.groups      = static_cast<uint8_t>(param.groups);
    m_param.strides[0]  = static_cast<uint8_t>(param.strides[0]); // x
    m_param.strides[1]  = static_cast<uint8_t>(param.strides[1]); // y
    m_param.dilation[0] = static_cast<uint8_t>(param.dilation[0]); // x
    m_param.dilation[1] = static_cast<uint8_t>(param.dilation[1]); // y
    m_param.padding[0]  = static_cast<uint8_t>(param.padding[0]); // t
    m_param.padding[1]  = static_cast<uint8_t>(param.padding[1]); // l
    m_param.padding[2]  = static_cast<uint8_t>(param.padding[2]); // b
    m_param.padding[3]  = static_cast<uint8_t>(param.padding[3]); // r
    m_param.kernel[0]   = static_cast<uint16_t>(m_weight.dim_at(0)); // oc
    m_param.kernel[1]   = static_cast<uint16_t>(m_weight.dim_at(1)); // ic
    m_param.kernel[2]   = static_cast<uint16_t>(m_weight.dim_at(2)); // kh
    m_param.kernel[3]   = static_cast<uint16_t>(m_weight.dim_at(3)); // kw
    return true;
}

bool Conv2dFunc::plan_forward_cpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (outputs.empty()) {
        outputs.push_back(Tensor(inputs[0].device()));
    }
    m_weight.reshape({m_param.kernel[0], m_param.kernel[1], m_param.kernel[2], m_param.kernel[3]});
    //  input dim order: [n, c, h, w]
    const int32_t ih = inputs[0].dim_at(2);
    const int32_t iw = inputs[0].dim_at(3);
    int32_t kh = m_param.dilation[1] * (m_weight.dim_at(2) - 1) + 1;
    int32_t kw = m_param.dilation[0] * (m_weight.dim_at(3) - 1) + 1;
    int32_t oh = ih < kh ? 0 : (ih+m_param.padding[0]+m_param.padding[2] - kh)/m_param.strides[1] +1;
    int32_t ow = iw < kw ? 0 : (iw+m_param.padding[1]+m_param.padding[3] - kw)/m_param.strides[0] +1;
    
    const int32_t oc = m_weight.dim_at(0);
    const int32_t in = inputs[0].dim_at(0);
    
    if (m_output_trans) {
        outputs[0].try_realloc({in, oh, ow, oc}, inputs[0].dtype());
    } else {
        outputs[0].try_realloc({in, oc, oh, ow}, inputs[0].dtype());
    }
    if (m_im2col.total_size() == 0) {
        m_im2col = Tensor(outputs[0].device());
    }
    m_im2col.try_realloc({inputs[0].dim_at(0), oh, ow, m_weight.stride_at(0)}, outputs[0].dtype());
    return true;
}

bool Conv2dFunc::_forward(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    // _parallel_sync(m_tp, outputs[0].total_size(), conv2d_element_split, std::ref(inputs[0]), std::ref(m_weight), std::ref(m_bias), std::ref(outputs[0]), m_param);
    _parallel_sync(m_tp, m_im2col.total_size(), im2col_element_split, std::ref(inputs[0]), std::ref(m_im2col), m_weight.dim_at(2), m_weight.dim_at(3), m_param.padding[0], m_param.padding[1], m_param.padding[2], m_param.padding[3], m_param.strides[1], m_param.strides[0], m_param.dilation[1], m_param.dilation[0], m_param.groups);
    // col.reshape({col.dim_at(0), col.dim_at(1)*col.dim_at(2), col.dim_at(3)});
    // auto wshape = m_weight.dims();
    // m_weight.reshape({m_weight.dim_at(0), m_weight.dim_at(1)*m_weight.dim_at(2)*m_weight.dim_at(3)});
    // Tensor per({inputs[0].dim_at(0), col.dim_at(1), m_weight.dim_at(0)});
    // per.mutable_ptr<float>();
    // _parallel_sync(m_tp, col.dim_at(0)*col.dim_at(1), matmul, std::ref(col),
    //                std::ref(m_weight), std::ref(m_bias), std::ref(per));
    // per.reshape({inputs[0].dim_at(0), outputs[0].dim_at(2), outputs[0].dim_at(3), outputs[0].dim_at(1)});
    // uint8_t perms[4] = {0, 3, 1, 2};
    // _parallel_sync(m_tp, per.total_size(), permute, std::ref(per), std::ref(outputs[0]), perms);
    // m_weight.reshape(wshape);
    if (true == m_output_trans) {
        m_weight.reshape({m_weight.dim_at(0), m_weight.dim_at(1)*m_weight.dim_at(2)*m_weight.dim_at(3)});
        m_im2col.reshape({m_im2col.dim_at(0), m_im2col.dim_at(1)*m_im2col.dim_at(2),
                m_im2col.dim_at(3)});
        _parallel_sync(m_tp, m_im2col.dim_at(0)*m_im2col.dim_at(1), matmul, std::ref(m_im2col),
                       std::ref(m_weight), std::ref(m_bias), std::ref(outputs[0]), 1.f, 1.f, m_act_cate);
    } else {
        m_weight.reshape({1, m_weight.dim_at(0), m_weight.dim_at(1)*m_weight.dim_at(2)*m_weight.dim_at(3)});
        for (int32_t n = 0; n < outputs[0].dim_at(0); ++n) {
            Tensor input({m_im2col.dim_at(1)*m_im2col.dim_at(2), m_im2col.dim_at(3)},
                         m_im2col.device(), m_im2col.unsafe_ptr<float>(n*m_im2col.stride_at(0)),
                         m_im2col.dtype());
            Tensor output({1, outputs[0].dim_at(1), outputs[0].dim_at(2)*outputs[0].dim_at(3)},
                          outputs[0].device(), outputs[0].unsafe_ptr<float>(n*outputs[0].stride_at(0)), outputs[0].dtype());
            _parallel_async(m_tp, m_weight.dim_at(0)*m_weight.dim_at(1), matmul, std::ref(m_weight),
                            input, std::ref(m_bias), output, 1.f, 1.f, m_act_cate);
        }
        m_tp->wait_work_complete();
    } // false == m_output_trans
    
    return true;
}

} // namespace mariana
