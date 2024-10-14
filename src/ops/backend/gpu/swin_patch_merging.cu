/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/swin_patch_merging.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-11:19:46:06
 * Description:
 * 
 */

#include <ops/swin_patch_merging.h>

#include <core/node.h>
#include <core/backend/gpu/cuda_common.h>
#include <core/backend/gpu/cuda_allocator.h>

namespace mariana {

void swin_patch_merge(SchedParam sched_param, const Tensor& input, Tensor& out, int32_t step, CUDAContext* cuda_ctx) {
    CudaIAllocator::CudaMemcoryContext cmc;
    cmc.stream = cuda_ctx->stream(sched_param.id_thread);
    cmc.kind   = cudaMemcpyHostToHost;
    cmc.sync   = true;
    const uint32_t IC = input.dim_at(3);
    IAllocator* allocator = get_allocator(out.device());
    for (uint32_t i = sched_param.this_thread_begin_index(); i < sched_param.this_thread_end_index(); ++i) {
        uint32_t idx = i;
        uint32_t idx4 = idx % 4; // c
        idx /= 4;
        uint32_t idx3 = idx % out.dim_at(2); // w
        idx /= out.dim_at(2);
        uint32_t idx2 = idx % out.dim_at(1); // h
        idx /= out.dim_at(1);
        uint32_t idx1 = idx; // n
        
        void* dst_data = out.unsafe_ptr<float>(i*IC);
        void* src_data = nullptr;
        allocator->memset(dst_data, 0, out.dtype().itemsize()*IC);
        switch (idx4) {
        case 0:
            if (idx2*step < (uint32_t)input.dim_at(1) && idx3*step < (uint32_t)input.dim_at(2)) {
                uint32_t input_offset = idx1*input.stride_at(0) + idx2*step*input.stride_at(1) +
                    idx3*step*input.stride_at(2);
                src_data = input.unsafe_ptr<float>(input_offset);
                allocator->memcpy(dst_data, src_data, input.dtype().itemsize()*IC, &cmc);
            }
            break;
        case 1:
            if (idx2*step+1 < (uint32_t)input.dim_at(1) && idx3*step < (uint32_t)input.dim_at(2)) {
                uint32_t input_offset = idx1*input.stride_at(0) + (idx2*step+1)*input.stride_at(1) +
                     idx3*step*input.stride_at(2);
                src_data = input.unsafe_ptr<float>(input_offset);
                allocator->memcpy(dst_data, src_data, input.dtype().itemsize()*IC, &cmc);
            }
            break;
        case 2:
            if (idx2*step < (uint32_t)input.dim_at(1) && idx3*step+1 < (uint32_t)input.dim_at(2)) {
                uint32_t input_offset = idx1*input.stride_at(0) + idx2*step*input.stride_at(1) +
                    (idx3*step+1)*input.stride_at(2);
                src_data = input.unsafe_ptr<float>(input_offset);
                allocator->memcpy(dst_data, src_data, input.dtype().itemsize()*IC, &cmc);
            }
            break;
        case 3:
            if ((idx2*step+1) < (uint32_t)input.dim_at(1) && (idx3*step+1) < (uint32_t)input.dim_at(2)) {
                uint32_t input_offset = idx1*input.stride_at(0) + (idx2*step+1)*input.stride_at(1) +
                    (idx3*step+1)*input.stride_at(2);
                src_data = input.unsafe_ptr<float>(input_offset);
                allocator->memcpy(dst_data, src_data, input.dtype().itemsize()*IC, &cmc);
            }
            break;
        default:
            MLOG(ERROR)<<"wrong number of swin patch merge to concat:"<<idx4;
            return;
        }
    }
    //cuda_ctx->stream_sync(cuda_ctx->stream(sched_param.id_thread));
}

bool SwinPatchMergingFunc::plan_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    cuda_set_device(cuda_ctx->device);
    if (outputs.empty()) {
        outputs.push_back(Tensor(DataOn::GPU));
    }
    
    int32_t oh = ceil(float(m_owner->info_shared_nodes()[0]->runtime_info().feature_height)/float(step));
    int32_t ow = ceil(float(m_owner->info_shared_nodes()[0]->runtime_info().feature_width)/float(step));
    int32_t oc = inputs[0].dim_size() == 3 ? inputs[0].dim_at(2) : inputs[0].dim_at(3);
    outputs[0].try_realloc({inputs[0].dim_at(0), oh, ow, oc*step*step}, inputs[0].dtype());
    return true;
}

bool SwinPatchMergingFunc::_forward_gpu(const tensor_list& inputs, tensor_list& outputs, ExeContext& context) {
    CUDAContext* cuda_ctx = static_cast<CUDAContext*>(m_owner->backend_ctx()->context);
    Tensor input = inputs[0].shallowcopy();
    input.reshape({input.dim_at(0), (int32_t)m_owner->info_shared_nodes()[0]->runtime_info().feature_height, (int32_t)m_owner->info_shared_nodes()[0]->runtime_info().feature_width, input.dim_at(2)});
    _parallel_sync(m_tp, outputs[0].total_size()/input.dim_at(3), swin_patch_merge, std::ref(input), std::ref(outputs[0]), step, cuda_ctx);
    return true;
}

} // namespace mariana
