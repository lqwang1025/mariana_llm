/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/roll.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-09:10:46:19
 * Description:
 * 
 */

#include <ops/backend/gpu/impl/roll.h>

namespace mariana {

template<typename T>
__global__ void __roll4_kernel(const T* input, T* out, int32_t idim0, int32_t idim1, int32_t idim2, int32_t idim3, uint32_t input_stride_0, uint32_t input_stride_1, uint32_t input_stride_2, uint32_t input_stride_3, int32_t dim1, int32_t dim2, int32_t shift1, int32_t shift2, int32_t odim0, int32_t odim1, int32_t odim2, int32_t odim3) {
    int32_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= idim0*idim1*idim2*idim3) return;
    int32_t idx = index;
    int32_t idx4 = idx % idim3;
    idx /= idim3;
    int32_t idx3 = idx % idim2;
    idx /= idim2;
    int32_t idx2 = idx % idim1;
    idx /= idim1;
    int32_t idx1 = idx;
    int32_t idxes[4] = {idx1, idx2, idx3, idx4};
    int32_t dims[2] = {dim1, dim2};
    int32_t shifts[2] = {shift1, shift2};
    int32_t odims[4] = {odim0, odim1, odim2, odim3};
    for (size_t i = 0; i < 2; ++i) {
        idxes[dims[i]] += shifts[i];
        if (odims[dims[i]] <= idxes[dims[i]]) {
            idxes[dims[i]] -= odims[dims[i]];
        } else if (idxes[dims[i]] < 0) {
            idxes[dims[i]] += odims[dims[i]];
        } else {
            // do nothing
        }
    }
    int32_t out_idx =
        idxes[0]*input_stride_0+idxes[1]*input_stride_1+
        idxes[2]*input_stride_2+idxes[3]*input_stride_3;
    out[out_idx] = input[index];
}

void roll4(SchedParam sched_param, const Tensor& input, Tensor& out, const RollParam& param, CUDAContext* cuda_ctx) {
    if (out.dtype().match<float>()) {
        uint32_t istride_0 = input.stride_at(0);
        uint32_t istride_1 = input.stride_at(1);
        uint32_t istride_2 = input.stride_at(2);
        uint32_t istride_3 = input.stride_at(3);
        const int32_t dim0 = input.dim_at(0);
        const int32_t dim1 = input.dim_at(1);
        const int32_t dim2 = input.dim_at(2);
        const int32_t dim3 = input.dim_at(3);
        float* input_ptr = input.unsafe_ptr<float>(0);
        float* out_ptr = out.unsafe_ptr<float>(0);
        __roll4_kernel<float><<<get_cuda_gridsize(input.total_size(), CUDA_ROLL_BLOCK_SIZE),
            CUDA_ROLL_BLOCK_SIZE, 0, cuda_ctx->stream(sched_param.id_thread)>>>(input_ptr, out_ptr, dim0, dim1, dim2, dim3, istride_0, istride_1, istride_2, istride_3, param.dims[0], param.dims[1], param.shifts[0], param.shifts[1], out.dim_at(0), out.dim_at(1), out.dim_at(2), out.dim_at(3));
        cuda_ctx->stream_sync(cuda_ctx->stream(sched_param.id_thread));
    } else {
        MLOG(FATAL)<<"roll4 unsupport datatype:"<<out.dtype().name();
    }

}

} // namespace mariana
