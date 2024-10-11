/*
 *        (C) COPYRIGHT Daniel Wang Limited.
 *             ALL RIGHTS RESERVED
 *
 * File       : ops/backend/gpu/impl/permute.cu
 * Authors    : lqwang@pandora
 * Create Time: 2024-10-09:13:27:39
 * Description:
 * 
 */

#include <ops/backend/gpu/impl/permute.h>

namespace mariana {

template<typename T>
__global__ void __permute4_kernel(const T* input, T* out, int32_t idim0, int32_t idim1, int32_t idim2, int32_t idim3, uint32_t o_stride_0, uint32_t o_stride_1, uint32_t o_stride_2, uint32_t o_stride_3, uint8_t perm0, uint8_t perm1, uint8_t perm2, uint8_t perm3) {
    int32_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= idim0*idim1*idim2*idim3) return;
    uint8_t perms[6] = {perm0, perm1, perm2, perm3};
    uint32_t idx = index;
    uint32_t idx4 = idx % idim3;
    idx /= idim3;
    uint32_t idx3 = idx % idim2;
    idx /= idim2;
    uint32_t idx2 = idx % idim1;
    idx /= idim1;
    uint32_t idx1 = idx;
    const uint32_t idxes[6] = {idx1, idx2, idx3, idx4};
    uint32_t dst_idx =
        idxes[perms[0]]*o_stride_0+idxes[perms[1]]*o_stride_1+
        idxes[perms[2]]*o_stride_2+idxes[perms[3]]*o_stride_3;
    out[dst_idx] = input[index];
}

void permute4(SchedParam sched_param, const Tensor& input, Tensor& out, uint8_t perms[4], CUDAContext* cuda_ctx) {
    if (out.dtype().match<float>()) {
        uint32_t istride_0 = input.stride_at(0);
        uint32_t istride_1 = input.stride_at(1);
        uint32_t istride_2 = input.stride_at(2);
        uint32_t istride_3 = input.stride_at(3);
        const int32_t dim0 = input.dim_at(0);
        const int32_t dim1 = input.dim_at(1);
        const int32_t dim2 = input.dim_at(2);
        const int32_t dim3 = input.dim_at(3);
        const uint32_t o_stride_0 = out.stride_at(0);
        const uint32_t o_stride_1 = out.stride_at(1);
        const uint32_t o_stride_2 = out.stride_at(2);
        const uint32_t o_stride_3 = out.stride_at(3);
        float* input_ptr = input.unsafe_ptr<float>(0);
        float* out_ptr = out.unsafe_ptr<float>(0);
        __permute4_kernel<float><<<get_cuda_gridsize(input.total_size(), CUDA_PERMUTE_BLOCK_SIZE),
            CUDA_PERMUTE_BLOCK_SIZE, 0, cuda_ctx->stream(sched_param.id_thread)>>>(input_ptr, out_ptr, dim0, dim1, dim2, dim3, o_stride_0, o_stride_1, o_stride_2, o_stride_3, perms[0], perms[1], perms[2], perms[3]);
        cuda_ctx->stream_sync(cuda_ctx->stream(sched_param.id_thread));
    } else {
        MLOG(FATAL)<<"permute6 unsupport datatype:"<<out.dtype().name();
    }
}

template<typename T>
__global__ void __permute6_kernel(const T* input, T* out, int32_t idim0, int32_t idim1, int32_t idim2, int32_t idim3, int32_t idim4, int32_t idim5, uint32_t o_stride_0, uint32_t o_stride_1, uint32_t o_stride_2, uint32_t o_stride_3, uint32_t o_stride_4, uint32_t o_stride_5, uint8_t perm0, uint8_t perm1, uint8_t perm2, uint8_t perm3, uint8_t perm4, uint8_t perm5) {
    int32_t index = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= idim0*idim1*idim2*idim3*idim4*idim5) return;
    uint8_t perms[6] = {perm0, perm1, perm2, perm3, perm4, perm5};
    uint32_t idx = index;
    uint32_t idx6 = idx % idim5;
    idx /= idim5;
    uint32_t idx5 = idx % idim4;
    idx /= idim4;
    uint32_t idx4 = idx % idim3;
    idx /= idim3;
    uint32_t idx3 = idx % idim2;
    idx /= idim2;
    uint32_t idx2 = idx % idim1;
    idx /= idim1;
    uint32_t idx1 = idx;
    const uint32_t idxes[6] = {idx1, idx2, idx3, idx4, idx5, idx6};
    uint32_t dst_idx =
        idxes[perms[0]]*o_stride_0+idxes[perms[1]]*o_stride_1+
        idxes[perms[2]]*o_stride_2+idxes[perms[3]]*o_stride_3+
        idxes[perms[4]]*o_stride_4+idxes[perms[5]]*o_stride_5;
    out[dst_idx] = input[index];
}

void permute6(SchedParam sched_param, const Tensor& input, Tensor& out, uint8_t perms[6], CUDAContext* cuda_ctx) {
    if (out.dtype().match<float>()) {
        uint32_t istride_0 = input.stride_at(0);
        uint32_t istride_1 = input.stride_at(1);
        uint32_t istride_2 = input.stride_at(2);
        uint32_t istride_3 = input.stride_at(3);
        const int32_t dim0 = input.dim_at(0);
        const int32_t dim1 = input.dim_at(1);
        const int32_t dim2 = input.dim_at(2);
        const int32_t dim3 = input.dim_at(3);
        const int32_t dim4 = input.dim_at(4);
        const int32_t dim5 = input.dim_at(5);
        const uint32_t o_stride_0 = out.stride_at(0);
        const uint32_t o_stride_1 = out.stride_at(1);
        const uint32_t o_stride_2 = out.stride_at(2);
        const uint32_t o_stride_3 = out.stride_at(3);
        const uint32_t o_stride_4 = out.stride_at(4);
        const uint32_t o_stride_5 = out.stride_at(5);
        float* input_ptr = input.unsafe_ptr<float>(0);
        float* out_ptr = out.unsafe_ptr<float>(0);
        __permute6_kernel<float><<<get_cuda_gridsize(input.total_size(), CUDA_PERMUTE_BLOCK_SIZE),
            CUDA_PERMUTE_BLOCK_SIZE, 0, cuda_ctx->stream(sched_param.id_thread)>>>(input_ptr, out_ptr, dim0, dim1, dim2, dim3, dim4, dim5, o_stride_0, o_stride_1, o_stride_2, o_stride_3, o_stride_4, o_stride_5, perms[0], perms[1], perms[2], perms[3], perms[4], perms[5]);
        cuda_ctx->stream_sync(cuda_ctx->stream(sched_param.id_thread));
    } else {
        MLOG(FATAL)<<"permute6 unsupport datatype:"<<out.dtype().name();
    }
}

} // namespace mariana
