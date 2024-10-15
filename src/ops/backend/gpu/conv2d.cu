/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/conv2d.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-07:21:38:45
 * Description:
 * 
 */

#include <ops/conv2d.h>
#include <ops/backend/gpu/impl/matmul.h>
#include <ops/backend/gpu/impl/im2col.h>

#include <core/node.h>
#include <core/backend/gpu/cuda_common.h>

namespace mariana {

bool Conv2dFunc::plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    if (inputs.size() != 1) {
        MLOG(ERROR)<<"Conv2d input's size must be 1 now:"<<inputs.size();
        return false;
    }
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    cuda_set_device(cuda_ctx->device);
    if (m_weight.device() != DataOn::GPU) {
        m_weight = m_weight.cuda(cuda_ctx->stream());;
    }
    m_weight.reshape({m_param.kernel[0], m_param.kernel[1], m_param.kernel[2], m_param.kernel[3]});
    if (m_bias.device() != DataOn::GPU) {
        m_bias = m_bias.cuda(cuda_ctx->stream());;
    }

    //  input dim order: [n, c, h, w]
    const int32_t ih = inputs[0].dim_at(2);
    const int32_t iw = inputs[0].dim_at(3);
    int32_t kh = m_param.dilation[1] * (m_weight.dim_at(2) - 1) + 1;
    int32_t kw = m_param.dilation[0] * (m_weight.dim_at(3) - 1) + 1;
    int32_t oh = ih < kh ? 0 : (ih+m_param.padding[0]+m_param.padding[2] - kh)/m_param.strides[1] +1;
    int32_t ow = iw < kw ? 0 : (iw+m_param.padding[1]+m_param.padding[3] - kw)/m_param.strides[0] +1;
    
    const int32_t oc = m_weight.dim_at(0);
    const int32_t in = inputs[0].dim_at(0);
    if (outputs.empty()) {
        outputs.push_back(Tensor(DataOn::GPU));
    }
    if (m_output_trans) {
        outputs[0].try_realloc({in, oh, ow, oc}, inputs[0].dtype());
    } else {
        outputs[0].try_realloc({in, oc, oh, ow}, inputs[0].dtype());
    }
    if (m_im2col.total_size() == 0) {
        m_im2col = Tensor(DataOn::GPU);
    }
    m_im2col.try_realloc({inputs[0].dim_at(0), oh, ow, m_weight.stride_at(0)}, outputs[0].dtype());
    return true;
}

bool Conv2dFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    _parallel_sync(m_tp, m_im2col.dim_at(0), im2col, std::ref(inputs[0]), std::ref(m_im2col), m_weight.dim_at(2), m_weight.dim_at(3), m_param.padding[0], m_param.padding[1], m_param.padding[2], m_param.padding[3], m_param.strides[1], m_param.strides[0], m_param.dilation[1], m_param.dilation[0], m_param.groups, cuda_ctx);
    if (true == m_output_trans) {
        m_weight.reshape({m_weight.dim_at(0), m_weight.dim_at(1)*m_weight.dim_at(2)*m_weight.dim_at(3)});
        m_im2col.reshape({m_im2col.dim_at(0), m_im2col.dim_at(1)*m_im2col.dim_at(2), m_im2col.dim_at(3)});
        Tensor output = outputs[0].shallowcopy();
        output.reshape({m_im2col.dim_at(0), m_im2col.dim_at(1), m_weight.dim_at(0)});
        _parallel_sync(m_tp, m_im2col.dim_at(0), matmul, std::ref(m_im2col), std::ref(m_weight), std::ref(m_bias), std::ref(output), 1.f, 1.f, m_act_cate, cuda_ctx);
    } else {
        m_weight.reshape({1, m_weight.dim_at(0), m_weight.dim_at(1)*m_weight.dim_at(2)*m_weight.dim_at(3)});
        for (int32_t n = 0; n < outputs[0].dim_at(0); ++n) {
            Tensor input({m_im2col.dim_at(1)*m_im2col.dim_at(2), m_im2col.dim_at(3)},
                         m_im2col.device(), m_im2col.unsafe_ptr<float>(n*m_im2col.stride_at(0)),
                         m_im2col.dtype());
            Tensor output({1, outputs[0].dim_at(1), outputs[0].dim_at(2)*outputs[0].dim_at(3)},
                          outputs[0].device(), outputs[0].unsafe_ptr<float>(n*outputs[0].stride_at(0)), outputs[0].dtype());
            _parallel_async(m_tp, m_weight.dim_at(0), matmul, std::ref(m_weight), input, std::ref(m_bias), output, 1.f, 1.f, m_act_cate, cuda_ctx);
        }
        m_tp->wait_work_complete();
    } // false == m_output_trans
    return true;
}

} // namespace mariana
