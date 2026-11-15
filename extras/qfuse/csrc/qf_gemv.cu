// qfuse: decode-size BF16 matrix-vector product for small projections of Qwen3.8-Flash-Next on SM120.
//   y[t, n] = sum_k x[t, k] * W[n, k]     x [T <= 8, K] bf16, W [N, K] bf16 (row-major), y [T, N] bf16
// One warp per output row: each lane owns K/32 consecutive elements, accumulates in fp32 and the warp sum is
// reduced with shuffles, so the summation order is fixed and results are deterministic.
// Measured in the served model this is NOT a win (see extras/qfuse/README.md): the tensor-core GEMMs it
// replaces look slow only because they share memory bandwidth with a large GEMV running beside them, and at
// 4 rows x 512 outputs this kernel is slower than the stock one. Kept as a small worked example of a qfuse
// kernel; off by default. K must be a multiple of 256 (8 elements per 16-byte load per lane).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>

namespace {

constexpr int MAXT = 8, WARPS = 8;

__device__ __forceinline__ float bf(uint32_t h) { return __uint_as_float(h << 16); }

__global__ void __launch_bounds__(WARPS * 32) gemv_bf16_kernel(
    const uint4 * __restrict__ x, const uint4 * __restrict__ w, __nv_bfloat16 * __restrict__ y, int T, int N, int K) {
    const int lane = threadIdx.x & 31, row = blockIdx.x * WARPS + (threadIdx.x >> 5);
    if (row >= N) return;
    const int per = K / 256;                 // 16-byte loads per lane
    const int k16 = K / 8;                   // 16-byte loads per row
    const uint4 * wr = w + (size_t) row * k16 + lane * per;
    float acc[MAXT];
#pragma unroll
    for (int t = 0; t < MAXT; ++t) acc[t] = 0.f;
    for (int i = 0; i < per; ++i) {
        const uint4 a = wr[i];
        const float w0 = bf(a.x & 0xffff), w1 = bf(a.x >> 16), w2 = bf(a.y & 0xffff), w3 = bf(a.y >> 16);
        const float w4 = bf(a.z & 0xffff), w5 = bf(a.z >> 16), w6 = bf(a.w & 0xffff), w7 = bf(a.w >> 16);
#pragma unroll
        for (int t = 0; t < MAXT; ++t) {
            if (t >= T) break;
            const uint4 b = x[(size_t) t * k16 + lane * per + i];
            float s = acc[t];
            s = fmaf(w0, bf(b.x & 0xffff), s); s = fmaf(w1, bf(b.x >> 16), s);
            s = fmaf(w2, bf(b.y & 0xffff), s); s = fmaf(w3, bf(b.y >> 16), s);
            s = fmaf(w4, bf(b.z & 0xffff), s); s = fmaf(w5, bf(b.z >> 16), s);
            s = fmaf(w6, bf(b.w & 0xffff), s); s = fmaf(w7, bf(b.w >> 16), s);
            acc[t] = s;
        }
    }
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
        if (t >= T) break;
        float s = acc[t];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        if (lane == 0) y[(size_t) t * N + row] = __float2bfloat16_rn(s);
    }
}

}  // namespace

at::Tensor gemv_bf16(const at::Tensor & x, const at::Tensor & w) {
    const int T = x.size(0), K = x.size(1), N = w.size(0);
    TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && T >= 1 && T <= MAXT && w.size(1) == K && K % 256 == 0);
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16);
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && x.is_cuda() && w.device() == x.device());
    TORCH_CHECK((reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0 && (reinterpret_cast<uintptr_t>(w.data_ptr()) & 15) == 0);
    const c10::cuda::CUDAGuard guard(x.device());
    auto y = at::empty({T, N}, x.options());
    gemv_bf16_kernel<<<(N + WARPS - 1) / WARPS, WARPS * 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint4 *>(x.data_ptr()), reinterpret_cast<const uint4 *>(w.data_ptr()),
        reinterpret_cast<__nv_bfloat16 *>(y.data_ptr()), T, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
