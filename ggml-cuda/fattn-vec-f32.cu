#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-vec-f32.cuh"

template<int D, int ncols, int parallel_blocks, vec_dot_KQ_f32_t vec_dot_KQ, bool Q_q8_1, dequantize_1_f32_t dequantize_1_v> // D == head size
#if !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
__launch_bounds__(D, 1)
#endif // !(defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__))
static __global__ void flash_attn_vec_ext_f32(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const int ne00,
        const int ne01,
        const int ne02,
        const int ne03,
        const int ne10,
        const int ne11,
        const int ne12,
        const int ne13,
        const int ne31,
        const int nb31,
        const int nb01,
        const int nb02,
        const int nb03,
        const int nb11,
        const int nb12,
        const int nb13,
        const int nb21,
        const int nb22,
        const int nb23,
        const int ne0,
        const int ne1,
        const int ne2,
        const int ne3) {
    //In this kernel Q, K, V are matrices while i, j, k are matrix indices.

    const int ic0 = (blockIdx.x / parallel_blocks) * ncols; // Index of the Q/QKV column to work on.
    const int ip  =  blockIdx.x % parallel_blocks; // Index in group of blocks running for the same column in parallel.

    const int gqa_ratio = ne02 / ne12; // With grouped query attention there are > 1 Q matrices per K, V matrix.
    const float  * Q_f   = (const float  *) (Q    + nb02* blockIdx.y              + nb01*ic0);
    const float2 * Q_f2  = (const float2 *)  Q_f;
    const char   * K_c   = (const char   *) (K    + nb12*(blockIdx.y / gqa_ratio));
    const char   * V_c   = (const char   *) (V    + nb22*(blockIdx.y / gqa_ratio)); // K and V have same shape
    const half   * maskh = (const half   *)  mask + ne11*ic0;

    const float slope = get_alibi_slope(max_bias, blockIdx.y, n_head_log2, m0, m1);

    static_assert(D % (2*WARP_SIZE) == 0, "D not divisible by 2*WARP_SIZE == 64.");
    constexpr int nwarps = D / WARP_SIZE;
    const int tid = WARP_SIZE*threadIdx.y + threadIdx.x;
    __builtin_assume(tid < D);

    __shared__ float KQ[ncols*D];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        KQ[j*D + tid] = -FLT_MAX/2.0f;
    }

    float kqmax[ncols];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqmax[j] = -FLT_MAX/2.0f;
    }
    float kqsum[ncols] = {0.0f};

    __shared__ float kqmax_shared[ncols][WARP_SIZE];
    __shared__ float kqsum_shared[ncols][WARP_SIZE];
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        if (threadIdx.y == 0) {
            kqmax_shared[j][threadIdx.x] = -FLT_MAX/2.0f;
            kqsum_shared[j][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    // Convert Q to half2 and store in registers:
    float2 Q_h2[ncols][D/(2*WARP_SIZE)];
    int    Q_i8[ncols][D/(sizeof(int)*QK8_1) == 0 ? 1 : D >= D/(sizeof(int)*QK8_1)];
    float2 Q_ds[ncols][D/QK8_1 == 0 ? 1 : D/QK8_1];
    if (Q_q8_1) {
#pragma unroll
        for (int j0 = 0; j0 < ncols; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

            int    * tmp_q_i8 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds = (float2 *) (tmp_q_i8 + D/sizeof(int));

            if (ncols > 2 && ic0 + j >= ne01) {
#pragma unroll
                for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                    const int i = i0 + threadIdx.x;

                    tmp_q_i8[i] = 0;
                }
                if (threadIdx.x < D/QK8_1) {
                    tmp_q_ds[threadIdx.x] = make_float2(0.0f, 0.0f);
                }
                continue;
            }

#pragma unroll
            for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                quantize_q8_1_to_shared<float2>(Q_f + j*(nb01/sizeof(float)) + 4*i0, scale, tmp_q_i8, tmp_q_ds);
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            int    * tmp_q_i8 = (int    *) &KQ[j*D];
            float2 * tmp_q_ds = (float2 *) (tmp_q_i8 + D/sizeof(int));

#pragma unroll
            for (int i0 = 0; i0 < D/sizeof(int); i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_i8[j][i0/WARP_SIZE] = tmp_q_i8[i];
                Q_ds[j][i0/WARP_SIZE] = tmp_q_ds[i/QI8_1];
            }
        }

        __syncthreads();
    } else {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
#pragma unroll
            for (int i0 = 0; i0 < D/2; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;

                Q_h2[j][i0/WARP_SIZE]    = ncols <= 2 || ic0 + j ? Q_f2[j*(nb01/sizeof(float2)) + i] : make_float2(0.0f, 0.0f);
                Q_h2[j][i0/WARP_SIZE].x *= scale;
                Q_h2[j][i0/WARP_SIZE].y *= scale;
            }
        }
    }

    float VKQ[ncols] = {0.0f};

    const int k_start = parallel_blocks == 1 ? 0 : ip*D;
    for (int k_VKQ_0 = k_start; k_VKQ_0 < ne11; k_VKQ_0 += parallel_blocks*D) {
        // Calculate KQ tile and keep track of new maximum KQ values:

        float kqmax_new_arr[ncols];
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            kqmax_new_arr[j] = kqmax[j];
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < D; i_KQ_0 += nwarps) {
            const int i_KQ = i_KQ_0 + threadIdx.y;

            if ((i_KQ_0 + nwarps > D && i_KQ >= D) || (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + i_KQ >= ne11)) {
                break;
            }

#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float sum = vec_dot_KQ(K_c + (k_VKQ_0 + i_KQ)*nb11, Q_h2[j], Q_i8[j], Q_ds[j]);
                sum = warp_reduce_sum(sum);
                sum += mask ? slope*__half2float(maskh[j*ne11 + k_VKQ_0 + i_KQ]) : 0.0f;

                kqmax_new_arr[j] = fmaxf(kqmax_new_arr[j], sum);

                if (threadIdx.x == 0) {
                    KQ[j*D + i_KQ] = sum;
                }
            }
        }

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            float kqmax_new_j = kqmax_new_arr[j];

            kqmax_new_j = warp_reduce_max(kqmax_new_j);
            if (threadIdx.x == 0) {
                kqmax_shared[j][threadIdx.y] = kqmax_new_j;
            }
        }

        __syncthreads();

#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            float kqmax_new_j = kqmax_shared[j][threadIdx.x];
            kqmax_new_j = warp_reduce_max(kqmax_new_j);

            const float KQ_max_scale = expf(kqmax[j] - kqmax_new_j);
            kqmax[j] = kqmax_new_j;

            const float val = expf(KQ[j*D + tid] - kqmax[j]);
            kqsum[j] = kqsum[j]*KQ_max_scale + val;
            KQ[j*D + tid] = val;

            VKQ[j] *= KQ_max_scale;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < D; ++k) {
            if (FATTN_KQ_STRIDE % D != 0 && k_VKQ_0 + k >= ne11) {
                break;
            }

            const float V_ki = dequantize_1_v(V_c + (k_VKQ_0 + k)*nb21, tid);
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                VKQ[j] += V_ki*KQ[j*D + k];
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        kqsum[j] = warp_reduce_sum(kqsum[j]);
        if (threadIdx.x == 0) {
            kqsum_shared[j][threadIdx.y] = kqsum[j];
        }
    }

    __syncthreads();

#pragma unroll
    for (int j_VKQ = 0; j_VKQ < ncols; ++j_VKQ) {
        if (ncols > 2 && ic0 + j_VKQ >= ne01) {
            break;
        }

        kqsum[j_VKQ] = kqsum_shared[j_VKQ][threadIdx.x];
        kqsum[j_VKQ] = warp_reduce_sum(kqsum[j_VKQ]);

        float dst_val = VKQ[j_VKQ];
        if (parallel_blocks == 1) {
            dst_val /= kqsum[j_VKQ];
        }
        const int j_dst = (ic0 + j_VKQ)*parallel_blocks + ip;
        dst[j_dst*D*gridDim.y + D*blockIdx.y + tid] = dst_val;
    }

    if (parallel_blocks != 1 && tid < ncols && (ncols <= 2 || ic0 + tid < ne01)) {
        dst_meta[(ic0 + tid)*gridDim.y*parallel_blocks + blockIdx.y*parallel_blocks + ip] = make_float2(kqmax[tid], kqsum[tid]);
    }
}

template <int D, int cols_per_block, int parallel_blocks, dequantize_1_f32_t dequantize_1_v>
void launch_fattn_tile_f32_K_type(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int nwarps = D/WARP_SIZE;
    const ggml_tensor * K = dst->src[1];

    switch (K->type) {
        case GGML_TYPE_Q4_0: {
            fattn_kernel_t fattn_kernel = flash_attn_vec_ext_f32<D, cols_per_block, parallel_blocks, vec_dot_fattn_vec_KQ_q4_0<float, D>, true, dequantize_1_v>;
            launch_fattn<D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block);
        } break;
        case GGML_TYPE_Q4_1: {
            fattn_kernel_t fattn_kernel = flash_attn_vec_ext_f32<D, cols_per_block, parallel_blocks, vec_dot_fattn_vec_KQ_q4_1<float, D>, true, dequantize_1_v>;
            launch_fattn<D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block);
        } break;
        case GGML_TYPE_Q5_0: {
            fattn_kernel_t fattn_kernel = flash_attn_vec_ext_f32<D, cols_per_block, parallel_blocks, vec_dot_fattn_vec_KQ_q5_0<float, D>, true, dequantize_1_v>;
            launch_fattn<D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block);
        } break;
        case GGML_TYPE_Q5_1: {
            fattn_kernel_t fattn_kernel = flash_attn_vec_ext_f32<D, cols_per_block, parallel_blocks, vec_dot_fattn_vec_KQ_q5_1<float, D>, true, dequantize_1_v>;
            launch_fattn<D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block);
        } break;
        case GGML_TYPE_Q8_0: {
            fattn_kernel_t fattn_kernel = flash_attn_vec_ext_f32<D, cols_per_block, parallel_blocks, vec_dot_fattn_vec_KQ_q8_0<float, D>, true, dequantize_1_v>;
            launch_fattn<D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block);
        } break;
        case GGML_TYPE_F16: {
            fattn_kernel_t fattn_kernel = flash_attn_vec_ext_f32<D, cols_per_block, parallel_blocks, vec_dot_fattn_vec_KQ_f16<float, D>, false, dequantize_1_v>;
            launch_fattn<D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block);
        } break;
        default:
            GGML_ASSERT(false);
            break;
    }
}

template <int cols_per_block, int parallel_blocks, dequantize_1_f32_t dequantize_1_v>
void launch_fattn_vec_f32_64_128(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    switch (Q->ne[0]) {
        case  64: {
            GGML_ASSERT(Q->type == GGML_TYPE_F16 && "Quantized K cache not supported for head size 64.");
            constexpr int D      = 64;
            constexpr int nwarps = D/WARP_SIZE;
            fattn_kernel_t fattn_kernel = flash_attn_vec_ext_f32<
                D, cols_per_block, parallel_blocks, vec_dot_fattn_vec_KQ_f16<float, D>, false, dequantize_1_v>;
            launch_fattn<D, parallel_blocks>(ctx, dst, fattn_kernel, nwarps, cols_per_block);
        } break;
        case 128: {
            constexpr int D      = 128;
            launch_fattn_tile_f32_K_type<D, cols_per_block, parallel_blocks, dequantize_1_v>(ctx, dst);
        } break;
        default: {
            GGML_ASSERT(false && "FlashAttention without tensor cores only supports head sizes 64 and 128.");
        } break;
    }
}

template <int cols_per_block, int parallel_blocks>
void launch_fattn_vec_f32_V_type(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * V = dst->src[2];

    switch (V->type) {
        case GGML_TYPE_Q4_0:
            launch_fattn_vec_f32_64_128<cols_per_block, parallel_blocks, dequantize_1_q4_0<float>>(ctx, dst);
            break;
        case GGML_TYPE_Q4_1:
            launch_fattn_vec_f32_64_128<cols_per_block, parallel_blocks, dequantize_1_q4_1<float>>(ctx, dst);
            break;
        case GGML_TYPE_Q5_0:
            launch_fattn_vec_f32_64_128<cols_per_block, parallel_blocks, dequantize_1_q5_0<float>>(ctx, dst);
            break;
        case GGML_TYPE_Q5_1:
            launch_fattn_vec_f32_64_128<cols_per_block, parallel_blocks, dequantize_1_q5_1<float>>(ctx, dst);
            break;
        case GGML_TYPE_Q8_0:
            launch_fattn_vec_f32_64_128<cols_per_block, parallel_blocks, dequantize_1_q8_0<float>>(ctx, dst);
            break;
        case GGML_TYPE_F16:
            launch_fattn_vec_f32_64_128<cols_per_block, parallel_blocks, dequantize_1_f16<float>>(ctx, dst);
            break;
        default:
            GGML_ASSERT(false);
            break;
    }
}

void ggml_cuda_flash_attn_ext_vec_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];

    if (Q->ne[1] == 1) {
        constexpr int cols_per_block  = 1;
        constexpr int parallel_blocks = 4;
        launch_fattn_vec_f32_V_type<cols_per_block, parallel_blocks>(ctx, dst);
        return;
    }

    if (Q->ne[1] == 2) {
        constexpr int cols_per_block  = 2;
        constexpr int parallel_blocks = 4;
        launch_fattn_vec_f32_V_type<cols_per_block, parallel_blocks>(ctx, dst);
        return;
    }

    if (Q->ne[1] <= 4) {
        constexpr int cols_per_block  = 4;
        constexpr int parallel_blocks = 4;
        launch_fattn_vec_f32_V_type<cols_per_block, parallel_blocks>(ctx, dst);
        return;
    }

    if (Q->ne[1] <= 8) {
        constexpr int cols_per_block  = 8;
        constexpr int parallel_blocks = 4;
        launch_fattn_vec_f32_V_type<cols_per_block, parallel_blocks>(ctx, dst);
        return;
    }

    constexpr int cols_per_block  = 8;
    constexpr int parallel_blocks = 1;
    launch_fattn_vec_f32_V_type<cols_per_block, parallel_blocks>(ctx, dst);
}
