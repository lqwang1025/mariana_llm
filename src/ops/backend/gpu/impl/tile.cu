/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/tile.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-10:05:50:17
 * Description:
 * 
 */

#include <ops/backend/gpu/impl/tile.h>
#include <core/backend/gpu/cuda_allocator.h>

namespace mariana {

void tile4(SchedParam sched_param, const Tensor& input, Tensor& out, uint32_t repeats[4], CUDAContext* cuda_ctx) {
    cuda_set_device(cuda_ctx->device);
    if (repeats[0] == 1 &&
        repeats[1] == 1 &&
        repeats[2] == 1 &&
        repeats[3] == 1 ) {
        out = input;
        return;
    }
    IAllocator* allocator = get_allocator(out.device());
    CudaIAllocator* cuda_alloc = static_cast<CudaIAllocator*>(allocator);
    CudaIAllocator::CudaMemcoryContext cmc;
    if (input.device() == DataOn::GPU) {
        cmc.stream = cuda_ctx->stream();
        cmc.sync = true;
        cmc.kind = cudaMemcpyDeviceToDevice;
    } else {
        MLOG(ERROR)<<"Unsupport device:"<<device_string(input.device());
        return;
    }
    const uint32_t repeat0 = repeats[0];
    const uint32_t repeat1 = repeats[1];
    const uint32_t repeat2 = repeats[2];
    const uint32_t repeat3 = repeats[3];
    const int32_t dim0 = input.dim_at(0);
    const int32_t dim1 = input.dim_at(1);
    const int32_t dim2 = input.dim_at(2);
    const int32_t dim3 = input.dim_at(3);
    if (out.dtype().match<float>()) {
        for (int32_t n = 0; n < dim0; ++n) {
            for (int32_t h = 0; h < dim1; ++h) {
                for (int32_t w = 0; w < dim2; ++w) {
                    const int32_t i_index = n*input.stride_at(0)+h*input.stride_at(1)+w*input.stride_at(2);
                    const int32_t o_index = n*out.stride_at(0)+h*out.stride_at(1)+w*out.stride_at(2);
                    
                    const float* ptr = input.unsafe_ptr<float>(i_index);
                    float* out_ptr = out.unsafe_ptr<float>(o_index);
                    for (int32_t p = 0; p < repeat3; ++p) {
                        cuda_alloc->memcpy(out_ptr, ptr, dim3*input.dtype().itemsize(), &cmc);
                        out_ptr += dim3;
                    }
                }
            }
            
            for (int32_t h = 0; h < dim1; ++h) {
                const int32_t i_index = n*out.stride_at(0)+h*out.stride_at(1);
                const int32_t o_index = n*out.stride_at(0)+h*out.stride_at(1)+dim2*out.stride_at(2);
                const float* ptr = out.unsafe_ptr<float>(i_index);
                float* out_ptr = out.unsafe_ptr<float>(o_index);
                const int32_t size = dim3*repeat3*dim2;
                for (int32_t p = 1; p < repeat2; ++p) {
                    cuda_alloc->memcpy(out_ptr, ptr, size*input.dtype().itemsize(), &cmc);
                    out_ptr += size;
                }
            }
            
            const int32_t i_index = n*out.stride_at(0);
            const int32_t o_index = n*out.stride_at(0)+dim1*out.stride_at(1);
            const float* ptr = out.unsafe_ptr<float>(i_index);
            float* out_ptr = out.unsafe_ptr<float>(o_index);
            const int32_t size = dim3*repeat3*dim2*repeat2*dim1;
            for (int32_t p = 1; p < repeat1; ++p) {
                cuda_alloc->memcpy(out_ptr, ptr, size*input.dtype().itemsize(), &cmc);
                out_ptr += size;
            }
        }
        const float* ptr = out.unsafe_ptr<float>(0);
        for (int32_t p = 1; p < repeat0; ++p) {
            float* out_ptr = out.unsafe_ptr<float>(p*out.stride_at(0));
            cuda_alloc->memcpy(out_ptr, ptr, out.total_size()*input.dtype().itemsize(), &cmc);
        }
        cuda_ctx->stream_sync(cuda_ctx->stream(sched_param.id_thread));
    } else {
        MLOG(FATAL)<<"tile4 unsupport datatype:"<<out.dtype().name();
    }
}

} // namespace mariana
