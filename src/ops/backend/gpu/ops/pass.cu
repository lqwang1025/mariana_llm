/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : pass.cu
 * Authors    : lqwang@inspur
 * Create Time: 2024-09-09:17:14:36
 * Description:
 * 
 */

#include <ops/pass.h>
#include <core/node.h>
#include <core/tensor_utils.h>
#include <core/backend/gpu/cuda_common.h>
#include <core/backend/gpu/cuda_allocator.h>

namespace mariana {

bool PassFunc::plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    cuda_set_device(cuda_ctx->device);
    Tensor input = inputs[0];
    if (outputs.empty()) {
        outputs.push_back(input.cuda(cuda_ctx->stream()));
    } else {
        outputs[0].try_realloc(input.dims(), input.dtype());
        IAllocator* allocator = get_allocator(outputs[0].device());
        CudaIAllocator* cuda_alloc = static_cast<CudaIAllocator*>(allocator);
        if (input.device() == DataOn::GPU) {
            CudaIAllocator::CudaMemcoryContext cmc;
            cmc.stream = cuda_ctx->stream();
            cmc.sync = false;
            cmc.kind = cudaMemcpyDeviceToDevice;
            cuda_alloc->memcpy(outputs[0].unsafe_ptr<uint8_t>(0), input.unsafe_ptr<uint8_t>(0), input.total_size()*input.dtype().itemsize(), &cmc);
        } else if (input.device() == DataOn::CPU) {
            CudaIAllocator::CudaMemcoryContext cmc;
            cmc.stream = cuda_ctx->stream();
            cmc.sync = false;
            cmc.kind = cudaMemcpyHostToDevice;
            cuda_alloc->memcpy(outputs[0].unsafe_ptr<uint8_t>(0), input.unsafe_ptr<uint8_t>(0), input.total_size()*input.dtype().itemsize(), &cmc);
        } else {
            MLOG(ERROR)<<"Unsupport device:"<<device_string(input.device());
            return false;
        }
    }
    return true;
}

bool PassFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    return true;
}

} // namespace mariana
