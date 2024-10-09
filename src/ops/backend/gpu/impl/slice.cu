/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/slice.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-10:05:50:17
 * Description:
 * 
 */

#include <ops/backend/gpu/impl/slice.h>

namespace mariana {

template<typename T>
__global__ void __slice4_kernel(const T* input, T* out, int32_t odim0, int32_t odim1, int32_t odim2, int32_t odim3, uint32_t input_stride_0, uint32_t input_stride_1, uint32_t input_stride_2, uint32_t input_stride_3, int32_t axis1, int32_t axis2, int32_t step1, int32_t step2, int32_t start1, int32_t start2) {
    int32_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= odim0*odim1*odim2*odim3) return;
    int32_t idx = index;
    int32_t idx4 = idx % odim3;
    idx /= odim3;
    int32_t idx3 = idx % odim2;
    idx /= odim2;
    int32_t idx2 = idx % odim1;
    idx /= odim1;
    int32_t idx1 = idx;
    int32_t idxes[4] = {idx1, idx2, idx3, idx4};
    int32_t axes[2] = {axis1, axis2};
    int32_t steps[2] = {step1, step2};
    int32_t starts[2] = {start1, start2};
    for (size_t i = 0; i < 2; ++i) {
        idxes[axes[i]] *= steps[i];
        idxes[axes[i]] += starts[i];
    }
    uint32_t input_idx =
        idxes[0]*input_stride_0+idxes[1]*input_stride_1+
        idxes[2]*input_stride_2+idxes[3]*input_stride_3;
    out[index] = input[input_idx];

}

void slice4(SchedParam sched_param, const Tensor& input, Tensor& out, const SliceParam& param, CUDAContext* cuda_ctx) {
    if (out.dtype().match<float>()) {
        if (param.axes.size() != 2) {
            MLOG(FATAL)<<"slice4 unsupport size:"<<param.axes.size();
        }
        uint32_t istride_0 = input.stride_at(0);
        uint32_t istride_1 = input.stride_at(1);
        uint32_t istride_2 = input.stride_at(2);
        uint32_t istride_3 = input.stride_at(3);
        const int32_t dim0 = out.dim_at(0);
        const int32_t dim1 = out.dim_at(1);
        const int32_t dim2 = out.dim_at(2);
        const int32_t dim3 = out.dim_at(3);
        float* input_ptr = input.unsafe_ptr<float>(0);
        float* out_ptr = out.unsafe_ptr<float>(0);
        __slice4_kernel<float><<<get_cuda_gridsize(input.total_size(), CUDA_SLICE_BLOCK_SIZE),
            CUDA_SLICE_BLOCK_SIZE, 0, cuda_ctx->stream(sched_param.id_thread)>>>(input_ptr, out_ptr, dim0, dim1, dim2, dim3, istride_0, istride_1, istride_2, istride_3, param.axes[0], param.axes[1], param.steps[0], param.steps[1], param.starts[0], param.starts[1]);
        cuda_ctx->stream_sync(cuda_ctx->stream(sched_param.id_thread));
    } else {
        MLOG(FATAL)<<"slice4 unsupport datatype:"<<out.dtype().name();
    }
}

} // namespace mariana
